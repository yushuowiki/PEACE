import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.autograd import Variable
from data_process import *

class EmbedMatcher_LSTMAE(nn.Module):
	def __init__(self, embed_dim, num_symbols, num_rel, num_ent, use_pretrain=True, embed=None, dropout=0.5, batch_size=64,
				 process_steps=4, finetune=False, aggregate='max',shot_K=3, cal_type = "train", max_neighbors = 50):
		super(EmbedMatcher_LSTMAE, self).__init__()
		self.embed_dim = embed_dim
		self.pad_idx = num_symbols

		self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
		self.aggregate = aggregate
		self.num_symbols = num_symbols
		self.layer_norm = LayerNormalization(2 * self.embed_dim)
		self.shot_K = shot_K
		self.max_neighbors = max_neighbors

		self.cal_type = cal_type

		self.gnn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
		self.gnn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

		self.dropout = nn.Dropout(dropout)

		self.set_rnn_encoder = nn.LSTM(2 * self.embed_dim, self.embed_dim, 2, bidirectional = True)
		self.set_rnn_decoder = nn.LSTM(2 * self.embed_dim, self.embed_dim, 2, bidirectional = True)

		self.set_FC_encoder = nn.Linear(3 * 2 * self.embed_dim, 2 * self.embed_dim)
		self.set_FC_decoder = nn.Linear(2 * self.embed_dim, 3 * 2 * self.embed_dim)

		# self.neigh_rnn = nn.LSTM(self.embed_dim, 50, 1, bidirectional = True)

		self.neigh_att_W = nn.Linear(2 * self.embed_dim, self.embed_dim)
		self.neigh_att_u = nn.Linear(self.embed_dim, 1)

		self.neigh_att_W_2 = nn.Linear(2 * self.embed_dim, self.embed_dim)
		self.neigh_att_u_2 = nn.Linear(self.embed_dim, 1)

		self.neigh_lin_W_first_layer = nn.Linear(2*self.embed_dim,self.embed_dim)
		self.neigh_lin_W_second_layer = nn.Linear(2 * self.embed_dim, self.embed_dim)

		self.set_att_W = nn.Linear(2 * self.embed_dim,1 * self.embed_dim)
		self.set_att_u = nn.Linear(1 * self.embed_dim, 1)

		self.bn = nn.BatchNorm1d(2 * self.embed_dim)
		self.softmax = nn.Softmax(dim=1)
		self.softmax_0 = nn.Softmax(dim=0)

		self.support_g_W = nn.Linear(4 * self.embed_dim, 2 * self.embed_dim)

		self.FC_query_g = nn.Linear(2 * self.embed_dim, 2 * self.embed_dim)
		self.FC_support_g_encoder = nn.Linear(2 * self.embed_dim, 2 * self.embed_dim)


		init.xavier_normal_(self.gnn_w.weight)
		init.xavier_normal_(self.neigh_att_W.weight)
		init.xavier_normal_(self.neigh_att_u.weight)
		init.xavier_normal_(self.set_att_W.weight)
		init.xavier_normal_(self.set_att_u.weight)
		init.xavier_normal_(self.support_g_W.weight)
		init.constant_(self.gnn_b, 0)

		init.xavier_normal_(self.FC_query_g.weight)
		init.xavier_normal_(self.FC_support_g_encoder.weight)

		if use_pretrain:
			self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
			if not finetune:
				self.symbol_emb.weight.requires_grad = False

		d_model = self.embed_dim * 2
		self.support_encoder = SupportEncoder(d_model, 2 * d_model, dropout)
		self.query_encoder = QueryEncoder(d_model, process_steps)

		self.support_output = nn.Linear(4*self.embed_dim,2*self.embed_dim)
		init.xavier_normal_(self.neigh_att_W_2.weight)
		init.xavier_normal_(self.neigh_att_u_2.weight)
		init.xavier_normal_(self.support_output.weight)

		self.path_att_W = nn.Linear(self.embed_dim, self.embed_dim)
		self.path_att_u = nn.Linear(self.embed_dim, 1)
		self.path_support_encoder_W = nn.Linear(self.embed_dim * 2, self.embed_dim)
		self.support_path_output = nn.Linear(self.embed_dim * 3, self.embed_dim * 2)

		self.GRUc = nn.GRUCell(input_size=self.embed_dim, hidden_size=self.embed_dim)
		self.GRUS = GRUS(self.GRUc, self.embed_dim, self.embed_dim, self.symbol_emb, "cuda")


	def neighbor_encoder(self,self_entity, connections, num_neighbors):

		self_entity = self_entity.cpu().numpy()
		self_entity = numpy.expand_dims(self_entity,1).repeat(self.max_neighbors,1)
		self_entity = Variable(torch.LongTensor(self_entity)).cuda()

		self_embeds = self.dropout(self.symbol_emb(self_entity))

		num_neighbors = num_neighbors.unsqueeze(1)

		relations = connections[:, :, 0].squeeze(-1)
		entities = connections[:, :, 1].squeeze(-1)

		rel_embeds = self.dropout(self.symbol_emb(relations)) 
		ent_embeds = self.dropout(self.symbol_emb(entities)) 

		concat_embeds_first_layer = torch.cat((self_embeds, rel_embeds), dim=-1)
		out_first_layer = F.leaky_relu(self.neigh_att_W(concat_embeds_first_layer))
		att_w_first_layer = self.neigh_att_u(out_first_layer)
		att_w_first_layer = self.softmax(att_w_first_layer).view(concat_embeds_first_layer.size()[0],1,self.max_neighbors)
		out_first_layer = torch.bmm(att_w_first_layer,concat_embeds_first_layer).view(concat_embeds_first_layer.size()[0],2*self.embed_dim)
		out_first_layer = self.neigh_lin_W_first_layer(out_first_layer)

		out_first_layer = out_first_layer.cpu().detach().numpy()
		out_first_layer = numpy.expand_dims(out_first_layer,1).repeat(self.max_neighbors,1)
		out_first_layer = Variable(torch.FloatTensor(out_first_layer)).cuda()

		concat_embeds_second_layer = torch.cat((out_first_layer, ent_embeds), dim=-1)
		out = F.leaky_relu(self.neigh_att_W_2(concat_embeds_second_layer))
		att_w = self.neigh_att_u_2(out)
		att_w = self.softmax(att_w).view(concat_embeds_second_layer.size()[0], 1, self.max_neighbors)

		att_w = torch.mul(att_w_first_layer,att_w)

		out = torch.bmm(att_w,concat_embeds_second_layer).view(concat_embeds_second_layer.size()[0], 2*self.embed_dim)
		out = self.neigh_lin_W_second_layer(out)

		return F.leaky_relu(out)


	def forward(self, query, support, support_rel, query_meta=None, support_meta=None, support_path = None, support_pair = None, support_path_entity = None, support_relation_set = None):


		query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
		support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

		query_left = self.neighbor_encoder(query[:,0],query_left_connections, query_left_degrees)
		query_right = self.neighbor_encoder(query[:,1],query_right_connections, query_right_degrees)

		support_left = self.neighbor_encoder(support[:,0],support_left_connections, support_left_degrees)
		support_right = self.neighbor_encoder(support[:,1],support_right_connections, support_right_degrees)

		query_neighbor = torch.cat((query_left, query_right), dim=-1) 
		support_neighbor = torch.cat((support_left, support_right), dim=-1)
		support = support_neighbor
		query = query_neighbor

		support_g = self.support_encoder(support)
		query_g = self.support_encoder(query)

		# lstm autoencoder
		support_g_0 = support_g.view(self.shot_K, 1, 2 * self.embed_dim)
		support_g_encoder, support_g_state = self.set_rnn_encoder(support_g_0)

		# lstm decoder for reconstruction loss
		support_g_decoder = support_g_encoder[-1].view(1, -1, 2 * self.embed_dim)
		decoder_set = []
		support_g_decoder_state = support_g_state
		for idx in range(self.shot_K):
			support_g_decoder, support_g_decoder_state = self.set_rnn_decoder(support_g_decoder,
																			  support_g_decoder_state)
			decoder_set.append(support_g_decoder)
		decoder_set = torch.cat(decoder_set, dim=0)

		ae_loss = nn.MSELoss()(support_g_0, decoder_set.detach())
		# ae_loss = 0

		# support_g_encoder = torch.mean(support_g_encoder, 0).view(1, 2*self.embed_dim)
		# support_g_encoder = support_g_encoder[-1].view(1, 2 * self.embed_dim)

		support_g_encoder = support_g_encoder.view(self.shot_K, 2 * self.embed_dim)

		support_g_encoder = support_g_0.view(self.shot_K, 2 * self.embed_dim) + support_g_encoder

		# support_g_encoder = torch.mean(support_g_encoder, dim=0, keepdim=True)

		support_g_att = self.set_att_W(support_g_encoder).tanh()
		att_w = self.set_att_u(support_g_att)
		att_w = self.softmax_0(att_w)
		support_g_encoder = torch.matmul(support_g_encoder.transpose(0, 1), att_w)
		support_g_encoder = support_g_encoder.transpose(0, 1)

		# rel_loss = self.tans_relation_encoder(support_left_connections,support_right_connections,support_g_encoder)

		support_g_encoder = support_g_encoder.view(1, 2 * self.embed_dim)

		num_T = max(map(len, support_rel))
		path_w = None

		if num_T > 0:
			support_tree_emb, support_tree_emb_list, t_h = self.GRUS(support_path, support_pair, support_path_entity, support_relation_set)
			if support_tree_emb != None:
				support_tree_emb_t = self.path_att_W(support_tree_emb).tanh()
				# path_out = F.relu(support_tree_emb_t)
				path_out = self.path_att_u(support_tree_emb_t)
				path_w = self.softmax_0(path_out)
				support_tree_emb = torch.matmul(support_tree_emb.transpose(0, 1), path_w)
				support_tree_emb = support_tree_emb.transpose(0, 1)
				support_g_encoder = self.support_path_output(torch.cat((support_g_encoder, support_tree_emb), dim=-1))
			else:
				print(1)

		query_f = self.query_encoder(support_g_encoder, query_g)


		# cosine similarity
		# query_g = self.FC_query_g(query_g)
		# support_g_encoder = self.FC_support_g_encoder(support_g_encoder)

		matching_scores = torch.matmul(query_f, support_g_encoder.t()).squeeze()

		return matching_scores, support_g_encoder, query_g, path_w, ae_loss


class LayerNormalization(nn.Module):
	''' Layer normalization module '''

	def __init__(self, d_hid, eps=1e-3):
		super(LayerNormalization, self).__init__()

		self.eps = eps
		self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
		self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

	def forward(self, z):
		if z.size(1) == 1:
			return z

		mu = torch.mean(z, keepdim=True, dim=-1)
		sigma = torch.std(z, keepdim=True, dim=-1)
		ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
		ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

		return ln_out


class SupportEncoder(nn.Module):
	"""docstring for SupportEncoder"""

	def __init__(self, d_model, d_inner, dropout=0.1):
		super(SupportEncoder, self).__init__()
		self.proj1 = nn.Linear(d_model, d_inner)
		self.proj2 = nn.Linear(d_inner, d_model)
		self.layer_norm = LayerNormalization(d_model)

		init.xavier_normal_(self.proj1.weight)
		init.xavier_normal_(self.proj2.weight)

		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()

	def forward(self, x):

		residual = x
		output = self.relu(self.proj1(x))
		output = self.dropout(self.proj2(output))
		return self.layer_norm(output + residual)


class QueryEncoder(nn.Module):
	def __init__(self, input_dim, process_step=4):
		super(QueryEncoder, self).__init__()
		self.input_dim = input_dim
		self.process_step = process_step
		self.process = nn.LSTMCell(input_dim, 2 * input_dim)

	def forward(self, support, query):
		'''
		support: (few, support_dim)
		query: (batch_size, query_dim)
		support_dim = query_dim
		return:
		(batch_size, query_dim)
		'''

		assert support.size()[1] == query.size()[1]

		if self.process_step == 0:
			return query

		batch_size = query.size()[0]
		h_r = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()
		c = Variable(torch.zeros(batch_size, 2 * self.input_dim)).cuda()

		# h_r = Variable(torch.zeros(batch_size, 2*self.input_dim))
		# c = Variable(torch.zeros(batch_size, 2*self.input_dim))

		for step in range(self.process_step):
			h_r_, c = self.process(query, (h_r, c))
			h = query + h_r_[:, :self.input_dim]  # (batch_size, query_dim)
			attn = F.softmax(torch.matmul(h, support.t()), dim=1)
			r = torch.matmul(attn, support)  # (batch_size, support_dim)
			h_r = torch.cat((h, r), dim=1)

		# return h_r_[:, :self.input_dim]
		return h


class GRUS(nn.Module):
	def __init__(self, GRUc, embed_dim, hidden_dim, symbol_emb, device):
		nn.Module.__init__(self)
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.symbol_emb = symbol_emb
		self.device = device
		self.hiddenRNN = GRUc
		self.cosSim = nn.CosineSimilarity(dim=1)
		self.w_d = nn.Linear(embed_dim, embed_dim)
		self.e_at_u = nn.Linear(embed_dim, 1)
		self.softmax_0 = nn.Softmax(dim=0)
		self.w_e = nn.Linear(embed_dim, embed_dim)
		self.w_p = nn.Linear(embed_dim, embed_dim)

	def forward(self, support_path, support_pair, support_path_entity, support_relation_set):
		support_pair_t = torch.tensor(support_pair)
		head_t = support_pair_t.select(-1, 0)
		tail_t = support_pair_t.select(-1, 1)
		head_t = head_t.to(self.device)
		tail_t = tail_t.to(self.device)
		t_h = self.symbol_emb(tail_t) - self.symbol_emb(head_t)

		support_path_filter = []
		support_path_entity_filter = []
		support_relation_set_filter = []

		flag = 0
		for i in range(len(support_path)):

			if i == 0 and len(support_path[i]) != 0:
				t_h_new = t_h[0].unsqueeze(0)
				support_path_filter.append(support_path[i])
				support_path_entity_filter.append(support_path_entity[i])
				support_relation_set_filter.append(support_relation_set[i])
				flag = 1

			if len(support_path[i]) != 0 and i != 0:
				support_path_filter.append(support_path[i])
				support_path_entity_filter.append(support_path_entity[i])
				support_relation_set_filter.append(support_relation_set[i])
				if flag == 1:
					t_h_new = torch.cat([t_h_new, t_h[i].unsqueeze(0)], dim=0)
				else:
					t_h_new = t_h[i].unsqueeze(0)
					flag = 1

		tail_emb = []
		symbol_pad = self.symbol_emb.weight.size()[0] - 1
		for i in range(len(support_path_filter)):
			support_relation_set_one = support_relation_set_filter[i]
			support_relation_set_one_emb = torch.tensor(support_relation_set_one).to(self.device)
			support_relation_set_one_emb = self.symbol_emb(support_relation_set_one_emb.long())
			exp_x = torch.exp(self.e_at_u(support_relation_set_one_emb))
			sum_exp_x = torch.sum(exp_x, dim=0, keepdim=True)

			tail_loc = torch.tensor(list(map(len, support_path_filter[i])), dtype=torch.long)

			support_path_one = support_path_filter[i]
			support_path_one_pad = list2tensor(support_path_one, padding_idx=symbol_pad, dtype=torch.int, device=self.device)
			# construct and initialize information loss
			cost_path_emb = torch.zeros(len(support_path_filter[i]), support_path_one_pad.size()[1], self.embed_dim).to(self.device)

			entity_tail_loc = torch.tensor(list(map(len, support_path_entity_filter[i])), dtype=torch.long)
			support_path_entity_one = support_path_entity_filter[i]
			support_path_entity_one_pad =list2tensor(support_path_entity_one, padding_idx=symbol_pad, dtype=torch.int, device=self.device)


			head_t_i = head_t.select(-1, i)
			head_t_i = head_t_i.repeat(support_path_entity_one_pad.size()[0], 1)
			head_t_i_emb = self.symbol_emb(head_t_i.long()).view(support_path_entity_one_pad.size()[0], self.embed_dim)

			tail_t_i = tail_t.select(-1, i)
			tail_t_i = tail_t_i.repeat(support_path_entity_one_pad.size()[0], 1)
			tail_t_i_emb = self.symbol_emb(tail_t_i.long()).view(support_path_entity_one_pad.size()[0], self.embed_dim)

			head_tail_sim_score = self.cosSim(head_t_i_emb, tail_t_i_emb)

			if support_path_one_pad.size()[1] == 1:
				one_step_rel = support_path_one_pad.select(-1, 0)
				sim_distance_rel_1 = (1 - head_tail_sim_score) / 2
				rel_1_emb = self.symbol_emb(one_step_rel.long())

				sim_distance_rel_1_emb = rel_1_emb * sim_distance_rel_1.unsqueeze(1)

				e_score_rel_1 = torch.exp(self.e_at_u(rel_1_emb)) / sum_exp_x
				e_score_rel_1_emb = rel_1_emb * e_score_rel_1

				d_out = torch.stack(tuple(sim_distance_rel_1_emb), dim=0).view(len(support_path_filter[i]), 1, self.embed_dim)
				e_out = torch.stack(tuple(e_score_rel_1_emb), dim=0).view(len(support_path_filter[i]), 1, self.embed_dim)
				d_out = self.w_d(d_out)
				e_out = self.w_e(e_out)
				c = F.relu(d_out + e_out)
				cost_path_emb[:, :1, :] = 0.001 * c

				lay = 1
			if support_path_one_pad.size()[1] == 2:
				one_step_rel = support_path_one_pad.select(-1, 0)
				two_step_rel = support_path_one_pad.select(-1, 1)

				entity_1 = support_path_one_pad.select(-1, 0)
				entity_1_emb = self.symbol_emb(entity_1.long())
				sim_entity_1 = (self.cosSim(head_t_i_emb, entity_1_emb) + self.cosSim(tail_t_i_emb, entity_1_emb)) / 2
				sim_rel_1 = (head_tail_sim_score + sim_entity_1) / 2
				sim_rel_2 = sim_rel_1
				sim_distance_rel_1 = (1 - sim_rel_1) / 2
				sim_distance_rel_2 = sim_distance_rel_1 / 2

				rel_1_emb = self.symbol_emb(one_step_rel.long())
				rel_2_emb = self.symbol_emb(two_step_rel.long())

				sim_distance_rel_1_emb = rel_1_emb * sim_distance_rel_1.unsqueeze(1)
				sim_distance_rel_2_emb = rel_2_emb * sim_distance_rel_2.unsqueeze(1)

				e_score_rel_1 = torch.exp(self.e_at_u(rel_1_emb)) / sum_exp_x
				e_score_rel_2 = torch.exp(self.e_at_u(rel_2_emb)) / sum_exp_x

				e_score_rel_1_emb = rel_1_emb * e_score_rel_1
				e_score_rel_2_emb = rel_2_emb * e_score_rel_2

				d_out = torch.stack((sim_distance_rel_1_emb, sim_distance_rel_2_emb), dim=1)
				e_out = torch.stack((e_score_rel_1_emb, e_score_rel_2_emb), dim=1)
				d_out = self.w_d(d_out)
				e_out = self.w_e(e_out)
				c = F.relu(d_out + e_out)
				cost_path_emb[:, :2, :] = 0.001 * c

				lay = 2
			if support_path_one_pad.size()[1] == 3:
				one_step_rel = support_path_one_pad.select(-1, 0)
				two_step_rel = support_path_one_pad.select(-1, 1)
				three_step_rel = support_path_one_pad.select(-1, 2)

				entity_1 = support_path_one_pad.select(-1, 0)
				entity_1_emb = self.symbol_emb(entity_1.long())
				entity_2 = support_path_one_pad.select(-1, 1)
				entity_2_emb = self.symbol_emb(entity_2.long())
				sim_entity_1 = (self.cosSim(head_t_i_emb, entity_1_emb) + self.cosSim(tail_t_i_emb, entity_1_emb)) / 2
				sim_entity_2 = (self.cosSim(head_t_i_emb, entity_2_emb) + self.cosSim(tail_t_i_emb, entity_2_emb)) / 2

				sim_rel_1 = (head_tail_sim_score + sim_entity_1) / 2
				sim_rel_2 = (sim_entity_1 + sim_entity_2) / 2
				sim_rel_3 = (head_tail_sim_score + sim_entity_2) / 2

				sim_distance_rel_1 = (1 - sim_rel_1) / 2
				sim_distance_rel_2 = (1 - sim_rel_2) / 2
				sim_distance_rel_3 = (1 - sim_rel_3) / 2

				rel_1_emb = self.symbol_emb(one_step_rel.long())
				rel_2_emb = self.symbol_emb(two_step_rel.long())
				rel_3_emb = self.symbol_emb(three_step_rel.long())

				sim_distance_rel_1_emb = rel_1_emb * sim_distance_rel_1.unsqueeze(1)
				sim_distance_rel_2_emb = rel_2_emb * sim_distance_rel_2.unsqueeze(1)
				sim_distance_rel_3_emb = rel_3_emb * sim_distance_rel_3.unsqueeze(1)

				e_score_rel_1 = torch.exp(self.e_at_u(rel_1_emb)) / sum_exp_x
				e_score_rel_2 = torch.exp(self.e_at_u(rel_2_emb)) / sum_exp_x
				e_score_rel_3 = torch.exp(self.e_at_u(rel_3_emb)) / sum_exp_x

				e_score_rel_1_emb = rel_1_emb * e_score_rel_1
				e_score_rel_2_emb = rel_2_emb * e_score_rel_2
				e_score_rel_3_emb = rel_3_emb * e_score_rel_3

				d_out = torch.stack((sim_distance_rel_1_emb, sim_distance_rel_2_emb, sim_distance_rel_3_emb), dim=1)
				e_out = torch.stack((e_score_rel_1_emb, e_score_rel_2_emb, e_score_rel_3_emb), dim=1)
				d_out = self.w_d(d_out)
				e_out = self.w_e(e_out)
				c = F.relu(d_out + e_out)
				cost_path_emb[:, :3, :] = 0.001 * c

				lay = 3

			if lay == 1:
				cost_path_emb_1 = cost_path_emb[:,0,:]
				cost_path_emb_1 = cost_path_emb_1.view(cost_path_emb.size()[0], self.embed_dim)
				one_step_rel_in = self.w_p(self.symbol_emb(one_step_rel.long()) - cost_path_emb_1)
				update_emb1 = self.hiddenRNN(one_step_rel_in)
			elif lay == 2:
				cost_path_emb_1 = cost_path_emb[:, 0, :]
				cost_path_emb_1 = cost_path_emb_1.view(cost_path_emb.size()[0], self.embed_dim)
				cost_path_emb_2 = cost_path_emb[:, 1, :]
				cost_path_emb_2 = cost_path_emb_2.view(cost_path_emb.size()[0], self.embed_dim)
				one_step_rel_in = self.w_p(self.symbol_emb(one_step_rel.long()) - cost_path_emb_1)
				two_step_rel_in = self.w_p(self.symbol_emb(two_step_rel.long()) - cost_path_emb_2)
				update_emb1 = self.hiddenRNN(one_step_rel_in)
				update_emb2 = self.hiddenRNN(two_step_rel_in, update_emb1)
			elif lay == 3:
				cost_path_emb_1 = cost_path_emb[:, 0, :]
				cost_path_emb_1 = cost_path_emb_1.view(cost_path_emb.size()[0], self.embed_dim)
				cost_path_emb_2 = cost_path_emb[:, 1, :]
				cost_path_emb_2 = cost_path_emb_2.view(cost_path_emb.size()[0], self.embed_dim)
				cost_path_emb_3 = cost_path_emb[:, 2, :]
				cost_path_emb_3 = cost_path_emb_3.view(cost_path_emb.size()[0], self.embed_dim)
				one_step_rel_in = self.w_p(self.symbol_emb(one_step_rel.long()) - cost_path_emb_1)
				two_step_rel_in = self.w_p(self.symbol_emb(two_step_rel.long()) - cost_path_emb_2)
				three_step_rel_in = self.w_p(self.symbol_emb(three_step_rel.long()) - cost_path_emb_3)
				update_emb1 = self.hiddenRNN(one_step_rel_in)
				update_emb2 = self.hiddenRNN(two_step_rel_in, update_emb1)
				update_emb3 = self.hiddenRNN(three_step_rel_in, update_emb2)
			batch = torch.arange(tail_loc.size()[0])
			tail_loc = tail_loc - 1

			if lay == 1:
				tail_emb_one = update_emb1[tail_loc]
			elif lay == 2:
				cat_emb = torch.cat([update_emb1.unsqueeze(1), update_emb2.unsqueeze(1)], dim=1)
				tail_emb_one = cat_emb[batch, tail_loc]
			elif lay == 3:
				cat_emb = torch.cat([update_emb1.unsqueeze(1), update_emb2.unsqueeze(1)], dim=1)
				cat_emb = torch.cat([cat_emb, update_emb3.unsqueeze(1)], dim=1)
				tail_emb_one = cat_emb[batch, tail_loc]
			tail_emb.append(tail_emb_one)

		tail_emb_all = None
		t_h_new = None
		for i in range(len(tail_emb)):
			if i == 0:
				tail_emb_all = tail_emb[0]
			else:
				tail_emb_all = torch.cat([tail_emb_all, tail_emb[i]], dim=0)

		if tail_emb_all == None:
			print(1)

		return tail_emb_all, tail_emb, t_h_new


class GRUS_1(nn.Module):
	def __init__(self, GRUc, embed_dim, hidden_dim, ent_emb, rel_emb, device):
		nn.Module.__init__(self)
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.ent_emb = ent_emb
		self.rel_emb = rel_emb
		self.device = device
		self.hiddenRNN = GRUc

	def forward(self, support_path, support_pair, support_path_entity, support_relation_set):
		support_pair_t = torch.tensor(support_pair)
		head_t = support_pair_t.select(-1, 0)
		tail_t = support_pair_t.select(-1, 1)
		head_t = head_t.to(self.device)
		tail_t = tail_t.to(self.device)
		t_h = self.ent_emb(tail_t) - self.ent_emb(head_t)

		support_path_filter = []
		flag = 0
		for i in range(len(support_path)):

			if i == 0 and len(support_path[i]) != 0:
				t_h_new = t_h[0].unsqueeze(0)
				support_path_filter.append(support_path[i])
				flag = 1
			
			if len(support_path[i]) != 0 and i != 0:
				support_path_filter.append(support_path[i])
				if flag == 1:
					t_h_new = torch.cat([t_h_new, t_h[i].unsqueeze(0)], dim=0)
				else:
					t_h_new = t_h[i].unsqueeze(0)
					flag = 1

		tail_emb = []
		rel_pad = self.rel_emb.weight.size()[0] - 1
		for i in range(len(support_path_filter)):

			tail_loc = torch.tensor(list(map(len, support_path_filter[i])), dtype=torch.long)

			support_path_one = support_path_filter[i]
			support_path_one_pad = list2tensor(support_path_one, padding_idx=rel_pad, dtype=torch.int,
											   device=self.device)
			if support_path_one_pad.size()[1] == 1:
				one_step_rel = support_path_one_pad.select(-1, 0)
				lay = 1
			if support_path_one_pad.size()[1] == 2:
				one_step_rel = support_path_one_pad.select(-1, 0)
				two_step_rel = support_path_one_pad.select(-1, 1)
				lay = 2
			if support_path_one_pad.size()[1] == 3:
				one_step_rel = support_path_one_pad.select(-1, 0)
				two_step_rel = support_path_one_pad.select(-1, 1)
				three_step_rel = support_path_one_pad.select(-1, 2)
				lay = 3

			if lay == 1:
				update_emb1 = self.hiddenRNN(self.rel_emb(one_step_rel.long()))
			elif lay == 2:
				update_emb1 = self.hiddenRNN(self.rel_emb(one_step_rel.long()))
				update_emb2 = self.hiddenRNN(self.rel_emb(two_step_rel.long()), update_emb1)
			elif lay == 3:
				update_emb1 = self.hiddenRNN(self.rel_emb(one_step_rel.long()))
				update_emb2 = self.hiddenRNN(self.rel_emb(two_step_rel.long()), update_emb1)
				update_emb3 = self.hiddenRNN(self.rel_emb(three_step_rel.long()), update_emb2)
			batch = torch.arange(tail_loc.size()[0])
			tail_loc = tail_loc - 1

			if lay == 1:
				tail_emb_one = update_emb1[tail_loc]
			elif lay == 2:
				cat_emb = torch.cat([update_emb1.unsqueeze(1), update_emb2.unsqueeze(1)], dim=1)
				tail_emb_one = cat_emb[batch, tail_loc]
			elif lay == 3:
				cat_emb = torch.cat([update_emb1.unsqueeze(1), update_emb2.unsqueeze(1)], dim=1)
				cat_emb = torch.cat([cat_emb, update_emb3.unsqueeze(1)], dim=1)
				tail_emb_one = cat_emb[batch, tail_loc]
			tail_emb.append(tail_emb_one)

		tail_emb_all = None
		t_h_new = None
		for i in range(len(tail_emb)):
			if i == 0:
				tail_emb_all = tail_emb[0]
			else:
				tail_emb_all = torch.cat([tail_emb_all, tail_emb[i]], dim=0)

		if tail_emb_all == None:
			print(1)

		return tail_emb_all, tail_emb, t_h_new


class GRUS_2(nn.Module):
	def __init__(self, GRUc, embed_dim, hidden_dim, ent_emb, rel_emb, device):
		nn.Module.__init__(self)
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.ent_emb = ent_emb
		self.rel_emb = rel_emb
		self.device = device
		self.hiddenRNN = GRUc
		self.cosSim = nn.CosineSimilarity(dim=0)
		self.w_d = nn.Linear(embed_dim, embed_dim)
		self.e_at_u = nn.Linear(embed_dim, 1)
		self.softmax_0 = nn.Softmax(dim=0)
		self.w_e = nn.Linear(embed_dim, embed_dim)

	def forward(self, support_path, support_pair, support_path_entity, support_relation_set):
		support_pair_t = torch.tensor(support_pair)
		head_t = support_pair_t.select(-1, 0)
		tail_t = support_pair_t.select(-1, 1)
		head_t = head_t.to(self.device)
		tail_t = tail_t.to(self.device)
		t_h = self.ent_emb(tail_t) - self.ent_emb(head_t)

		support_path_filter = []
		support_path_entity_filter = []
		support_relation_set_filter = []

		flag = 0
		for i in range(len(support_path)):

			if i == 0 and len(support_path[i]) != 0:
				t_h_new = t_h[0].unsqueeze(0)
				support_path_filter.append(support_path[i])
				support_path_entity_filter.append(support_path_entity[i])
				support_relation_set_filter.append(support_relation_set[i])
				flag = 1

			if len(support_path[i]) != 0 and i != 0:
				support_path_filter.append(support_path[i])
				support_path_entity_filter.append(support_path_entity[i])
				support_relation_set_filter.append(support_relation_set[i])
				if flag == 1:
					t_h_new = torch.cat([t_h_new, t_h[i].unsqueeze(0)], dim=0)
				else:
					t_h_new = t_h[i].unsqueeze(0)
					flag = 1

		tail_emb = []
		rel_pad = self.rel_emb.weight.size()[0] - 1
		for i in range(len(support_path_filter)):
			head_entity_emb = self.ent_emb(head_t[i])
			tail_entity_emb = self.ent_emb(tail_t[i])
			sim_head_tail = self.cosSim(head_entity_emb, tail_entity_emb)

			relation_att_dict = {}
			support_relation_set_one = support_relation_set_filter[i]
			support_relation_set_one_emb = torch.tensor(support_relation_set_one).to(self.device)
			support_relation_set_one_emb = self.rel_emb(support_relation_set_one_emb.long())

			support_relation_set_att = self.softmax_0(self.e_at_u(support_relation_set_one_emb))
			for index in range(len(support_relation_set_one)):
				relation_att_dict[support_relation_set_one[index]] = support_relation_set_att[index]

			tail_loc = torch.tensor(list(map(len, support_path_filter[i])), dtype=torch.long)
			support_path_one = support_path_filter[i]
			support_path_one_pad = list2tensor(support_path_one, padding_idx=rel_pad, dtype=torch.int, device=self.device)
			cost_path_emb = torch.zeros(len(support_path_filter[i]), support_path_one_pad.size()[1], self.embed_dim).to(self.device)

			for j in range(len(support_path_filter[i])):

				path_entity_one = torch.tensor(support_path_entity_filter[i][j])
				path_entity_one = path_entity_one.to(self.device)

				path_relation_one = torch.tensor(support_path_one[j]).to(self.device)
				path_relation_one_emb = self.rel_emb(path_relation_one)

				if len(path_entity_one) == 0:
					sim_distance_entity_1 = 1 - sim_head_tail
					distance_entity_1_emb = sim_distance_entity_1 * path_relation_one_emb
					d_out = self.w_d(distance_entity_1_emb)
					rel_1_att = relation_att_dict[support_path_filter[i][j][0]]
					e_out = self.w_e(path_relation_one_emb * rel_1_att)
					c = F.leaky_relu(d_out + e_out)
					cost_path_emb[j, :1, :] = c

				if len(path_entity_one) == 1:
					rel_1_att = relation_att_dict[support_path_filter[i][j][0]]
					rel_2_att = relation_att_dict[support_path_filter[i][j][1]]

					first_relation_emb = path_relation_one_emb.select(0, 0)
					second_relation_emb = path_relation_one_emb.select(0, 1)
					e_out = torch.stack((first_relation_emb * rel_1_att, second_relation_emb * rel_2_att), dim=0)
					e_out = self.w_e(e_out)

					first_entity_emb = self.ent_emb(path_entity_one.select(0, 0))
					sim_entity_1 = (self.cosSim(head_entity_emb, first_entity_emb) + self.cosSim(tail_entity_emb,
																								 first_entity_emb)) / 2
					sim_distance_entity_1 = 1 - (sim_head_tail + sim_entity_1) / 2
					sim_distance_entity_2 = sim_distance_entity_1
					sim_distance_entity_1_emb = sim_distance_entity_1 * path_relation_one_emb.select(0, 0)
					sim_distance_entity_2_emb = sim_distance_entity_2 * path_relation_one_emb.select(0, 1)
					distance_entity_emb = torch.stack((sim_distance_entity_1_emb, sim_distance_entity_2_emb), dim=0)
					d_out = self.w_d(distance_entity_emb)
					c = F.leaky_relu(d_out + e_out)

					cost_path_emb[j, :2, :] = c

				if len(path_entity_one) == 2:
					rel_1_att = relation_att_dict[support_path_filter[i][j][0]]
					rel_2_att = relation_att_dict[support_path_filter[i][j][1]]
					rel_3_att = relation_att_dict[support_path_filter[i][j][2]]

					first_relation_emb = path_relation_one_emb.select(0, 0)
					second_relation_emb = path_relation_one_emb.select(0, 1)
					third_relation_emb = path_relation_one_emb.select(0, 2)

					e_out = torch.stack((first_relation_emb * rel_1_att, second_relation_emb * rel_2_att,
										 third_relation_emb * rel_3_att), dim=0)
					e_out = self.w_e(e_out)

					first_entity_emb = self.ent_emb(path_entity_one.select(0, 0))
					second_entity_emb = self.ent_emb(path_entity_one.select(0, 1))
					sim_entity_1 = (self.cosSim(head_entity_emb, first_entity_emb) + self.cosSim(tail_entity_emb,
																								 first_entity_emb)) / 2
					sim_entity_2 = (self.cosSim(head_entity_emb, second_entity_emb) + self.cosSim(tail_entity_emb,
																								  second_entity_emb)) / 2

					sim_distance_entity_1 = 1 - (sim_head_tail + sim_entity_1) / 2
					sim_distance_entity_2 = 1 - (sim_entity_1 + sim_entity_2) / 2
					sim_distance_entity_3 = 1 - (sim_entity_2 + sim_head_tail) / 2

					sim_distance_entity_1_emb = sim_distance_entity_1 * path_relation_one_emb.select(0, 0)
					sim_distance_entity_2_emb = sim_distance_entity_2 * path_relation_one_emb.select(0, 1)
					sim_distance_entity_3_emb = sim_distance_entity_3 * path_relation_one_emb.select(0, 2)

					distance_entity_emb = torch.stack(
						(sim_distance_entity_1_emb, sim_distance_entity_2_emb, sim_distance_entity_3_emb), dim=0)
					d_out = self.w_d(distance_entity_emb)
					c = F.leaky_relu(d_out + e_out)

					cost_path_emb[j, :3, :] = c

			cost_path_emb = 0.001 * cost_path_emb

			if support_path_one_pad.size()[1] == 1:
				one_step_rel = support_path_one_pad.select(-1, 0)
				lay = 1
			if support_path_one_pad.size()[1] == 2:
				one_step_rel = support_path_one_pad.select(-1, 0)
				two_step_rel = support_path_one_pad.select(-1, 1)
				lay = 2
			if support_path_one_pad.size()[1] == 3:
				one_step_rel = support_path_one_pad.select(-1, 0)
				two_step_rel = support_path_one_pad.select(-1, 1)
				three_step_rel = support_path_one_pad.select(-1, 2)
				lay = 3

			if lay == 1:
				cost_path_emb_1 = cost_path_emb[:, 0, :]
				cost_path_emb_1 = cost_path_emb_1.view(cost_path_emb.size()[0], self.embed_dim)
				update_emb1 = self.hiddenRNN(self.rel_emb(one_step_rel.long()) - cost_path_emb_1)
			elif lay == 2:
				cost_path_emb_1 = cost_path_emb[:, 0, :]
				cost_path_emb_1 = cost_path_emb_1.view(cost_path_emb.size()[0], self.embed_dim)
				cost_path_emb_2 = cost_path_emb[:, 1, :]
				cost_path_emb_2 = cost_path_emb_2.view(cost_path_emb.size()[0], self.embed_dim)
				update_emb1 = self.hiddenRNN(self.rel_emb(one_step_rel.long()) - cost_path_emb_1)
				update_emb2 = self.hiddenRNN(self.rel_emb(two_step_rel.long()) - cost_path_emb_2, update_emb1)
			elif lay == 3:
				cost_path_emb_1 = cost_path_emb[:, 0, :]
				cost_path_emb_1 = cost_path_emb_1.view(cost_path_emb.size()[0], self.embed_dim)
				cost_path_emb_2 = cost_path_emb[:, 1, :]
				cost_path_emb_2 = cost_path_emb_2.view(cost_path_emb.size()[0], self.embed_dim)
				cost_path_emb_3 = cost_path_emb[:, 2, :]
				cost_path_emb_3 = cost_path_emb_3.view(cost_path_emb.size()[0], self.embed_dim)
				update_emb1 = self.hiddenRNN(self.rel_emb(one_step_rel.long()) - cost_path_emb_1)
				update_emb2 = self.hiddenRNN(self.rel_emb(two_step_rel.long()) - cost_path_emb_2, update_emb1)
				update_emb3 = self.hiddenRNN(self.rel_emb(three_step_rel.long()) - cost_path_emb_3, update_emb2)
			batch = torch.arange(tail_loc.size()[0])
			tail_loc = tail_loc - 1

			if lay == 1:
				tail_emb_one = update_emb1[tail_loc]
			elif lay == 2:
				cat_emb = torch.cat([update_emb1.unsqueeze(1), update_emb2.unsqueeze(1)], dim=1)
				tail_emb_one = cat_emb[batch, tail_loc]
			elif lay == 3:
				cat_emb = torch.cat([update_emb1.unsqueeze(1), update_emb2.unsqueeze(1)], dim=1)
				cat_emb = torch.cat([cat_emb, update_emb3.unsqueeze(1)], dim=1)
				tail_emb_one = cat_emb[batch, tail_loc]
			tail_emb.append(tail_emb_one)

		tail_emb_all = None
		t_h_new = None
		for i in range(len(tail_emb)):
			if i == 0:
				tail_emb_all = tail_emb[0]
			else:
				tail_emb_all = torch.cat([tail_emb_all, tail_emb[i]], dim=0)

		if tail_emb_all == None:
			print(1)

		return tail_emb_all, tail_emb, t_h_new