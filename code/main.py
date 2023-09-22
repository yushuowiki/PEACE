from typing import Dict, Any
import logging
import numpy as np
import random
import torch
from args import read_args
import data_process
import json
from data_generator import *
from matcher import *
from matcher_lstmae import *
import torch.nn.functional as F
from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import os
torch.set_num_threads(1)
os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.backends.cudnn.enabled = False

class Model_Run(object):
	def __init__(self, arg):
		super(Model_Run, self).__init__()
		for k, v in vars(arg).items():
			setattr(self, k, v)

		self.device = torch.device("cuda")
		self.meta = not self.no_meta
		self.cuda = arg.cuda

		if self.random_embed:
			use_pretrain = False
		else:
			use_pretrain = True

		logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
		if self.test or self.random_embed:
			self.load_symbol2id()
			use_pretrain = False
		else:
			# load pretrained embedding
			self.load_embed()

		self.num_symbols = len(self.symbol2id.keys()) - 1 # one for 'PAD'
		self.pad_id = self.num_symbols
		self.use_pretrain = use_pretrain
		self.set_aggregator = args.set_aggregator
		self.embed_dim = args.embed_dim
		self.max_neighbor = args.max_neighbor

		if self.set_aggregator == 'lstmae':
			self.matcher = EmbedMatcher_LSTMAE(self.embed_dim, self.num_symbols, len(self.rel2id), len(self.ent2id), use_pretrain=self.use_pretrain,
											   embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size,
											   process_steps=self.process_steps, finetune=self.fine_tune,
											   aggregate=self.aggregator,shot_K=args.few, max_neighbors=self.max_neighbor)
		else:
			self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain,
										embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size,
										process_steps=self.process_steps, finetune=self.fine_tune,
										aggregate=self.aggregator)

		if self.cuda:
			self.matcher.cuda()

		self.batch_nums = 0
		self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())

		self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
		self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10000], gamma=0.25)

		self.ent2id = json.load(open(self.datapath + '/ent2ids'))
		self.num_ents = len(self.ent2id.keys())

		logging.info('BUILDING CONNECTION MATRIX')
		degrees = self.build_graph(max_=self.max_neighbor)

		logging.info('LOADING CANDIDATES ENTITIES')
		self.rel2candidates = json.load(open(self.datapath + '/rel2candidates_all.json')) 

		self.e1rel_e2 = defaultdict(list)
		self.e1rel_e2 = json.load(open(self.datapath + '/e1rel_e2.json'))


	def load_symbol2id(self):      
		symbol_id = {}
		rel2id = json.load(open(self.datapath + '/relation2ids'))
		ent2id = json.load(open(self.datapath + '/ent2ids'))
		i = 0
		for key in rel2id.keys():
			if key not in ['','OOV']:
				symbol_id[key] = i
				i += 1
		self.rel2id = rel2id

		for key in ent2id.keys():
			if key not in ['', 'OOV']:
				symbol_id[key] = i
				i += 1
		self.ent2id = ent2id

		symbol_id['PAD'] = i
		self.symbol2id = symbol_id
		self.symbol2vec = None

		path_dict_str = json.load(open(self.datapath + '/train_valid_test_pair2paths_name_no_inv_with_nodes.json'))
		self.trian_test_path, self.train_test_path_id, self.train_test_path_entity, self.train_test_path_entity_id = path_read(path_dict_str, symbol_id)


	def load_embed(self):
		symbol_id = {}
		rel_id_symbol_id = {}
		ent_id_symbol_id = {}
		rel2id = json.load(open(self.datapath + '/relation2ids'))
		ent2id = json.load(open(self.datapath + '/ent2ids'))

		logging.info('LOADING PRE-TRAINED EMBEDDING')
		if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
			ent_embed = np.loadtxt(self.datapath + '/embed/entity2vec.' + self.embed_model)
			rel_embed = np.loadtxt(self.datapath + '/embed/relation2vec.' + self.embed_model)

			if self.embed_model == 'ComplEx':
				# normalize the complex embeddings
				ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
				ent_std = np.std(ent_embed, axis=1, keepdims=True)
				rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
				rel_std = np.std(rel_embed, axis=1, keepdims=True)
				eps = 1e-3
				ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
				rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

			assert ent_embed.shape[0] == len(ent2id.keys())
			assert rel_embed.shape[0] == len(rel2id.keys())

			i = 0
			embeddings = []
			for key in rel2id.keys():
				if key not in ['','OOV']:
					symbol_id[key] = i
					rel_id_symbol_id[rel2id[key]] = i
					i += 1
					embeddings.append(list(rel_embed[rel2id[key],:]))

			self.rel2id = rel2id

			for key in ent2id.keys():
				if key not in ['', 'OOV']:
					symbol_id[key] = i
					ent_id_symbol_id[ent2id[key]] = i
					i += 1
					embeddings.append(list(ent_embed[ent2id[key],:]))
			self.ent2id = ent2id

			symbol_id['PAD'] = i
			embeddings.append(list(np.zeros((rel_embed.shape[1],))))
			embeddings = np.array(embeddings)
			assert embeddings.shape[0] == len(symbol_id.keys())

			self.symbol2id = symbol_id
			self.symbol2vec = embeddings
			self.relid2symbolid = rel_id_symbol_id
			self.entid2symbolid = ent_id_symbol_id

			train_tasks = json.load(open(self.datapath + '/train_tasks.json'))
			test_tasks = json.load(open(self.datapath + '/test_tasks.json'))
			dev_tasks = json.load(open(self.datapath + '/dev_tasks.json'))

			train_rel = list(train_tasks.keys())
			test_rel = list(test_tasks.keys())
			dev_rel = list(dev_tasks.keys())

			rel2id1 = {}
			ent2id1 = {}
			rel_embedding = []
			ent_embedding = []
			i = 0
			for key in rel2id.keys():
				rel2id1[key] = i
				i = i + 1
				rel_embedding.append(list(rel_embed[rel2id[key], :]))

			for rel in list(train_rel) + list(test_rel) + list(dev_rel):
				rel2id1[rel] = i
				i = i + 1
			j = 0
			for key in ent2id.keys():
				ent2id1[key] = j
				j = j + 1
				ent_embedding.append(list(ent_embed[ent2id[key], :]))

			rel_embedding = torch.tensor(rel_embedding)
			ent_embedding = torch.tensor(ent_embedding)

			self.rel_emb = nn.Embedding(len(rel2id1.keys()) + 1, self.embed_dim)
			self.rel_emb.weight.data[:len(rel2id)] = rel_embedding
			self.rel_emb.weight.data[-1] = torch.zeros(1, 100)
			self.rel_emb = self.rel_emb.to(self.device)

			self.ent_emb = nn.Embedding(len(ent2id.keys()) + 1, self.embed_dim)
			self.ent_emb.weight.data[:len(ent2id)] = ent_embedding
			self.ent_emb.weight.data[-1] = torch.zeros(1, 100)
			self.ent_emb = self.ent_emb.to(self.device)

			path_dict_str = json.load(open(self.datapath + '/train_valid_test_pair2paths_name_no_inv_with_nodes.json'))
			self.trian_test_path, self.train_test_path_id, self.train_test_path_entity, self.train_test_path_entity_id = path_read(path_dict_str, symbol_id)





	def build_graph(self, max_=50):
		self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
		self.e1_rele2 = defaultdict(list)
		self.e1_degrees = defaultdict(int)

		with open(self.datapath + '/path_graph') as f:
			lines = f.readlines()
			print(lines[0].rstrip().split())
			for line in tqdm(lines):
				e1,rel,e2 = line.rstrip().split()
				self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
				self.e1_rele2[e2].append((self.symbol2id[rel+'_inv'], self.symbol2id[e1]))

		degrees = {}
		for ent, id_ in self.ent2id.items():
			neighbors = self.e1_rele2[ent]
			if len(neighbors) > max_:
				neighbors = neighbors[:max_]
			# degrees.append(len(neighbors)) 
			degrees[ent] = len(neighbors)
			self.e1_degrees[id_] = len(neighbors) # add one for self conn
			for idx, _ in enumerate(neighbors):
				self.connections[id_, idx, 0] = _[0]
				self.connections[id_, idx, 1] = _[1]

		json.dump(degrees, open(self.datapath + '/degrees', 'w'))
		# assert 1==2
		for degree in degrees:
			print("degree1:",degree,degrees[degree])
			break
		return degrees


	def data_analysis(self):
		#data_process.rel_triples_dis(self.datapath)
		#data_process.build_vocab(self.datapath)
		data_process.candidate_triples(self.datapath)
		print("data analysis finish")


	def get_meta(self, left, right):
		if self.cuda:
			left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).cuda()
			left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
			right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).cuda()
			right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
		else:
			left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0)))
			left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left]))
			right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0)))
			right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right]))

		return (left_connections, left_degrees, right_connections, right_degrees)


	def save(self, path=None):
		if not path:
			path = self.save_path
		torch.save(self.matcher.state_dict(), path)


	def load(self):
		self.matcher.load_state_dict(torch.load(self.save_path))


	def train(self):
		logging.info('START TRAINING...')
		best_hits10 = 0.0
		hits10_file = open(self.datapath + "_hits10.txt", "w")
		hits5_file = open(self.datapath + "_hits5.txt", "w")
		hits1_file = open(self.datapath + "_hits1.txt", "w")
		mrr_file = open(self.datapath + "_mrr.txt", "w")

		for data in train_generate(self.datapath, self.batch_size, self.few, self.symbol2id, self.ent2id, self.e1rel_e2):
			# 获取某一任务的支持集，查询集，消极集
			support_pairs, support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data


			# 获取三元组实体对的connection情况和领域节点数量degree
			support_meta = self.get_meta(support_left, support_right)
			query_meta = self.get_meta(query_left, query_right)
			false_meta = self.get_meta(false_left, false_right)

			for support_pair in support_pairs:
				support_pair[0] = self.symbol2id[support_pair[0]]
				support_pair[1] = self.symbol2id[support_pair[1]]


			support_rel = rel_submit(support_pairs, self.train_test_path_id)
			support_path = path_submit(support_pairs, self.train_test_path_id)
			support_path_entity = ent_submit(support_pairs, self.train_test_path_entity_id)

			if self.cuda:
				support = Variable(torch.LongTensor(support)).cuda()
				query = Variable(torch.LongTensor(query)).cuda()
				false = Variable(torch.LongTensor(false)).cuda()
			else:
				support = Variable(torch.LongTensor(support))
				query = Variable(torch.LongTensor(query))
				false = Variable(torch.LongTensor(false))

			if self.no_meta:
				if self.set_aggregator == 'lstmae':
					query_scores, ae_loss = self.matcher(query, support)
					false_scores, ae_loss = self.matcher(false, support)
				else:
					query_scores = self.matcher(query, support)
					false_scores = self.matcher(false, support)
			else:
				if self.set_aggregator == 'lstmae':
					query_scores, support_g_embed, query_g_embed, path_w, ae_loss = self.matcher(query, support, support_rel, query_meta, support_meta, support_path, support_pairs, support_path_entity, support_rel)
					write_path_attention(path_w)
					false_scores, support_g_embed, query_g_embed, path_w, ae_loss = self.matcher(false, support, support_rel, false_meta, support_meta, support_path, support_pairs, support_path_entity, support_rel)
					write_path_attention(path_w)
				else:
					query_scores, query_g_embed = self.matcher(query, support, query_meta, support_meta)
					false_scores, query_g_embed = self.matcher(false, support, false_meta, support_meta)

			# calculate loss through relu activation function
			margin_ = query_scores - false_scores
			loss = F.relu(self.margin - margin_).mean()

			if self.set_aggregator == 'lstmae':
				loss += args.ae_weight * ae_loss


			self.optim.zero_grad()
			loss.backward()

			try:
				self.optim.step()
			except RuntimeError as exception:
				if "out of memory" in str(exception):
					print("WARNING: out of memory")
					if hasattr(torch.cuda, 'empty_cache'):
						torch.cuda.empty_cache()
				else:
					raise exception
			if self.batch_nums % self.eval_every == 0:
				logging.info('batch num: '+str(self.batch_nums))
				logging.info('loss: '+str(loss))

				hits10, hits5, hits1, mrr = self.eval(meta=self.meta)

				hits10_file.write(str(("%.3f" % hits10)) + "\n")
				hits5_file.write(str(("%.3f" % hits5)) + "\n")
				hits1_file.write(str(("%.3f" % hits1)) + "\n")
				mrr_file.write(str(("%.3f" % mrr)) + "\n")

				self.save()

				if hits10 > best_hits10:
					self.save(self.save_path + '_bestHits10')
					best_hits10 = hits10

			self.batch_nums += 1

			self.scheduler.step()
			if self.batch_nums == self.max_batches:
				self.save()
				break
				hits10_file.close()
				hits5_file.close()
				hits1_file.close()
				mrr_file.close()


	def eval(self, mode='test', meta=False):
		self.matcher.eval()

		symbol2id = self.symbol2id
		few = self.few

		logging.info('EVALUATING ON %s DATA' % mode.upper())
		if mode == 'dev':
			test_tasks = json.load(open(self.datapath + '/dev_tasks.json'))
		else:
			test_tasks = json.load(open(self.datapath + '/test_tasks.json'))

		rel2candidates = self.rel2candidates

		hits10 = []
		hits5 = []
		hits1 = []
		mrr = []

		task_embed_f = open(self.datapath + "task_embed.txt", "w")

		temp_count = 0
		for query_ in test_tasks.keys():
			#print (query_)
			entity_embed_f = open(self.datapath + str(query_) + "_entity_embed.txt", "w")
			
			task_embed_f.write(str(query_) + ",")

			hits10_ = []
			hits5_ = []
			hits1_ = []
			mrr_ = []

			candidates = rel2candidates[query_]

			support_triples = test_tasks[query_][:few]

			temp_count += 1

			support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

			support_rel = rel_submit(support_pairs, self.train_test_path_id)
			support_path = path_submit(support_pairs, self.train_test_path_id)
			support_path_entity = ent_submit(support_pairs, self.train_test_path_entity_id)

			if meta:
				support_left = [self.ent2id[triple[0]] for triple in support_triples]
				support_right = [self.ent2id[triple[2]] for triple in support_triples]
				support_meta = self.get_meta(support_left, support_right)

			if self.cuda:
				support = Variable(torch.LongTensor(support_pairs)).cuda()
			else:
				support = Variable(torch.LongTensor(support_pairs))

			temp = 0

			for triple in test_tasks[query_][few:]:
				temp += 1
				true = triple[2]
				query_pairs = []
				if triple[0] in symbol2id and triple[2] in symbol2id:
					query_pairs.append([symbol2id[triple[0]], symbol2id[triple[2]]])

				if meta:
					query_left = []
					query_right = []
					if triple[0] in self.ent2id and triple[2] in self.ent2id:
						query_left.append(self.ent2id[triple[0]])
						query_right.append(self.ent2id[triple[2]])

				for ent in candidates:
					if (ent not in self.e1rel_e2[triple[0]+triple[1]]) and ent != true:
						query_pairs.append([symbol2id[triple[0]], symbol2id[ent]])
						if meta:
							query_left.append(self.ent2id[triple[0]])
							query_right.append(self.ent2id[ent])

				if self.cuda:
					query = Variable(torch.LongTensor(query_pairs)).cuda()
				else:
					query = Variable(torch.LongTensor(query_pairs))

				if meta:
					query_meta = self.get_meta(query_left, query_right)
					if self.set_aggregator == 'lstmae':
						scores, support_g_embed, query_g_embed, path_w, _ = self.matcher(query, support, support_rel, query_meta, support_meta, support_path, support_pairs, support_path_entity, support_rel)

						write_path_attention(path_w)


						#print (support_g_embed.cpu().data.numpy()[:10])
						support_g_embed = support_g_embed.view(support_g_embed.numel())
						query_g_embed_temp = query_g_embed.cpu().detach().numpy()
						#query_g_embed_temp = query_g_embed.cpu().data.numpy()
						if temp == 1:
							for k in range(1, len(query_g_embed_temp)):
								#query_g_embed_temp = query_g_embed[k].view(query_g_embed[k].numel())
								#query_g_embed_temp = query_g_embed_temp.cpu().data.numpy()
								for l in range(len(query_g_embed_temp[k])):
									entity_embed_f.write(str(query_g_embed_temp[k][l]) + " ")
								entity_embed_f.write("\n")
						
						#query_g_embed_temp = query_g_embed[0].view(query_g_embed[0].numel())
						#query_g_embed_temp = query_g_embed_temp.cpu().data.numpy()
						for l in range(len(query_g_embed_temp[0])):
							entity_embed_f.write(str(query_g_embed_temp[0][l]) + " ")
						entity_embed_f.write("\n")

						if temp == 1:
							embed_temp = support_g_embed.cpu().data.numpy()
							#print (len(embed_temp))
							for l in range(len(embed_temp)):
								task_embed_f.write(str(float(embed_temp[l])) + " ")
							task_embed_f.write("\n")

					else:
						scores, query_g_embed = self.matcher(query, support, query_meta, support_meta)
						query_g_embed_temp = query_g_embed.cpu().detach().numpy()
						#query_g_embed_temp = query_g_embed.cpu().data.numpy()
						if temp == 1:
							for k in range(1, len(query_g_embed_temp)):
								#query_g_embed_temp = query_g_embed[k].view(query_g_embed[k].numel())
								#query_g_embed_temp = query_g_embed_temp.cpu().data.numpy()
								for l in range(len(query_g_embed_temp[k])):
									entity_embed_f.write(str(query_g_embed_temp[k][l]) + " ")
								entity_embed_f.write("\n")
						
						#query_g_embed_temp = query_g_embed[0].view(query_g_embed[0].numel())
						#query_g_embed_temp = query_g_embed_temp.cpu().data.numpy()
						for l in range(len(query_g_embed_temp[0])):
							entity_embed_f.write(str(query_g_embed_temp[0][l]) + " ")
						entity_embed_f.write("\n")

						# if temp == 1:
						# 	embed_temp = support_g_embed.cpu().data.numpy()
						# 	#print (len(embed_temp))
						# 	for l in range(len(embed_temp)):
						# 		task_embed_f.write(str(float(embed_temp[l])) + " ")
						# 	task_embed_f.write("\n")

					scores.detach()
					scores = scores.data
				else:
					if self.set_aggregator == 'lstmae':
						scores, _ = self.matcher(query, support)
					else:
						scores = self.matcher(query, support)
					scores.detach()
					scores = scores.data

				scores = scores.cpu().numpy()
				sort = list(np.argsort(scores))[::-1]
				rank = sort.index(0) + 1

				if rank <= 10:
					hits10.append(1.0)
					hits10_.append(1.0)
				else:
					hits10.append(0.0)
					hits10_.append(0.0)
				if rank <= 5:
					hits5.append(1.0)
					hits5_.append(1.0)
				else:
					hits5.append(0.0)
					hits5_.append(0.0)
				if rank <= 1:
					hits1.append(1.0)
					hits1_.append(1.0)
				else:
					hits1.append(0.0)
					hits1_.append(0.0)
				mrr.append(1.0/rank)
				mrr_.append(1.0/rank)

			logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(query_, np.mean(hits10_), np.mean(hits5_), np.mean(hits1_), np.mean(mrr_)))
			logging.info('Number of candidates: {}, number of text examples {}'.format(len(candidates), len(hits10_)))

		logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
		logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
		logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
		logging.critical('MRR: {:.3f}'.format(np.mean(mrr)))
		task_embed_f.close()
		entity_embed_f.close()

		self.matcher.train()

		return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)

	def test_(self):
		self.load()
		logging.info('Pre-trained model loaded')
		self.eval(mode='dev', meta=self.meta)
		self.eval(mode='test', meta=self.meta)

if __name__ == '__main__':
	args = read_args()
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

	fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	
	logger.addHandler(ch)
	logger.addHandler(fh)

	# setup random seeds
	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed)

	# model execution 
	model_run = Model_Run(args)

	# data analysis
	#model_run.data_analysis()
	
	# train/test model
	if args.test_:
		model_run.test_()
	else:
		model_run.train()
	#
	# model_run.train()



