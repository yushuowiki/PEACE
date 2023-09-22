import numpy as np
from collections import defaultdict
import json
import torch
import os

# distribution of relation triples
def build_vocab(datapath):
	rels = set()
	ents = set()

	with open(datapath + '/path_graph') as f:
		lines = f.readlines()
		for line in lines:
			line = line.rstrip()
			rel = line.split('\t')[1]
			e1 = line.split('\t')[0]
			e2 = line.split('\t')[2]
			rels.add(rel)
			rels.add(rel + '_inv')
			ents.add(e1)
			ents.add(e2)
	
	# relation/entity id map		
	relationid = {}
	for idx, item in enumerate(list(rels)):
		relationid[item] = idx

	entid = {}
	for idx, item in enumerate(list(ents)):
		entid[item] = idx

	#print len(entid)

	json.dump(relationid, open(datapath + '/relation2ids', 'w'))
	json.dump(entid, open(datapath + '/ent2ids', 'w'))  


def candidate_triples(datapath):
	ent2ids = json.load(open(datapath+'/ent2ids'))

	all_entities = ent2ids.keys()

	type2ents = defaultdict(set)
	for ent in all_entities:
		try:
			type_ = ent.split(':')[1]
			type2ents[type_].add(ent)
		except Exception as e:
			continue

	train_tasks = json.load(open(datapath + '/known_rels.json'))
	dev_tasks = json.load(open(datapath + '/dev_tasks.json'))
	test_tasks = json.load(open(datapath + '/test_tasks.json'))

	all_reason_relations = train_tasks.keys() + dev_tasks.keys() + test_tasks.keys()

	all_reason_relation_triples = train_tasks.values() + dev_tasks.values() + test_tasks.values()
	
	assert len(all_reason_relations) == len(all_reason_relation_triples) 

	rel2candidates = {}
	for rel, triples in zip(all_reason_relations, all_reason_relation_triples):
		possible_types = set()
		for example in triples:
			try:
				type_ = example[2].split(':')[1] # type of tail entity
				possible_types.add(type_)
			except Exception as e:
				print (example)

		candidates = []
		for type_ in possible_types:
			candidates += list(type2ents[type_])

		candidates = list(set(candidates))
		if len(candidates) > 1000:
			candidates = candidates[:1000]
		rel2candidates[rel] = candidates

		#rel2candidates[rel] = list(set(candidates))
		
	json.dump(rel2candidates, open(datapath + '/rel2candidates_all.json', 'w'))


def for_filtering(datapath, save=False):
	e1rel_e2 = defaultdict(list)
	train_tasks = json.load(open(datapath + '/train_tasks.json'))
	dev_tasks = json.load(open(datapath + '/dev_tasks.json'))
	test_tasks = json.load(open(datapath + '/test_tasks.json'))
	few_triples = []
	for _ in (train_tasks.values() + dev_tasks.values() + test_tasks.values()):
		few_triples += _
	for triple in few_triples:
		e1,rel,e2 = triple
		e1rel_e2[e1+rel].append(e2)
	if save:
		json.dump(e1rel_e2, open(datapath + '/e1rel_e2.json', 'w'))



def set_rel_sim_count(num_rel_id):
	sim_set = {}
	for key, item1 in num_rel_id.items():
		list_rel_all = item1
		set_rel_all = []
		for i in range(len(list_rel_all)):
			set_rel_all.append(set(list_rel_all[i]))
		set_onerel_sim = []
		set_avg = []
		for setm in set_rel_all:
			set_one_sim = []
			for setn in set_rel_all:
				bing = list(set(setm) | set(setn))
				jiao = list(set(setm)&set(setn))
				if len(bing)!=0:
					set_one_sim.append(len(jiao)/len(bing))
				else:
					set_one_sim.append(0)
			set_onerel_sim.append(set_one_sim)
			sum_one = (sum(set_one_sim) - 1)/(len(set_one_sim)-1)
			set_avg.append(sum_one)
		sim_set[key] = set_avg
	return sim_set

def test_relkind(rel_test_trip, train_test_path_id):
	test2relkind_dict = {}
	test2relkind = {}
	for key, item in rel_test_trip.items():
		test2relkind_dict[key] = []
		test2relkind[key] = []
		for i in range(len(item)):
			trip = item[i]
			pair = (trip[0],trip[2])
			path = train_test_path_id[pair]
			path_set = set()
			for m in range(len(path)):
				for n in range(len(path[m])):
					path_set.add(path[m][n])
			test2relkind_dict[key].append(pair)
			test2relkind[key].append(list(path_set))
	return test2relkind_dict, test2relkind

def relation_item2id(item, symbol2id):
	item_str = []
	item_id = []
	for i in range(len(item)):
		str = []
		one = []
		for j in range(int((len(item[i]) + 1)/2)):
			one.append(symbol2id[item[i][j * 2]])
			str.append(item[i][j * 2])
		item_id.append(one)
		item_str.append(str)
	return item_str, item_id

def entity_item2id(item, symbol2id):
	item_str = []
	item_id = []
	for i in range(len(item)):
		str = []
		one = []
		for j in range(int((len(item[i]) - 1)/2)):
			one.append(symbol2id[item[i][j * 2 + 1]])
			str.append(item[i][j * 2 + 1])
		item_id.append(one)
		item_str.append(str)
	return item_str, item_id

def path_read(path_dict_str, symbol2id):
	# construct the node sequence on a path
	path_entity_str = {}
	path_entity_id = {}

	path_relation_str = {}
	path_relation_id = {}
	for key, item in path_dict_str.items():
		h, t = key.split('&')[0], key.split('&')[1]
		key_update = (h, t)
		key_update_id = (symbol2id[h], symbol2id[t])
		if len(item) != 0:
			relation_item_str, relation_item_id = relation_item2id(item, symbol2id)
			entity_item_str, entity_item_id = entity_item2id(item, symbol2id)
		else:
			relation_item_str = []
			relation_item_id = []
			entity_item_str = []
			entity_item_id = []
		path_relation_str[key_update] = relation_item_str
		path_relation_id[key_update_id] = relation_item_id
		path_entity_str[key_update] = entity_item_str
		path_entity_id[key_update_id] = entity_item_id
	return path_relation_str, path_relation_id, path_entity_str, path_entity_id

def pad_tensor(tensor: torch.Tensor, length, value=0, dim=0) -> torch.Tensor:
	return torch.cat(
		(tensor, tensor.new_full((*tensor.size()[:dim], length - tensor.size(dim), *tensor.size()[dim + 1:]), value)),
		dim=dim)

def list2tensor(data_list: list, padding_idx,  dtype=torch.long,  device=torch.device("cpu")):
	max_len = max(map(len, data_list))
	max_len = max(max_len, 1)
	data_tensor = torch.stack(
		tuple(pad_tensor(torch.tensor(data, dtype=dtype), max_len, padding_idx, 0) for data in data_list)).to(device)
	return data_tensor

def rel_submit(pair,train_test_path_id):
	rel_all = []
	for i in range(len(pair)):
		pair_one = (pair[i][0], pair[i][1])
		rel_list = train_test_path_id[pair_one]
		rel_set = set()
		for m in range(len(rel_list)):
			for n in range(len(rel_list[m])):
				rel_set.add(rel_list[m][n])
		rel_all.append(list(rel_set))
	return rel_all


def path_submit(pair, train_test_path_id):
	path_all = []
	for i in range(len(pair)):
		pair_one = (pair[i][0], pair[i][1])
		path_list = train_test_path_id[pair_one]
		path_all.append(path_list)
	return path_all

def ent_submit(pair,train_test_entity_id):
	entity_all = []
	for i in range(len(pair)):
		pair_one = (pair[i][0], pair[i][1])
		entity_list = train_test_entity_id[pair_one]
		entity_all.append(entity_list)
	return entity_all



def write_path_attention(path_attention):
	if path_attention != None and path_attention.size()[0] > 20:
		path_attention_file = open("../path_attention.txt", "a")
		path_attention_file.write(str(path_attention) + "\n")
		path_attention_file.close()


def write_relation_attention(relation_attention):
	relation_attention_file = open("../relation_attention.txt", "w")
	relation_attention_file.write(str(relation_attention) + "\n")
	relation_attention_file.close()


def path_subgraph(dataset):

	ent2id = json.load(open(os.path.join(dataset, 'ent2ids')))
	rel2id = json.load(open(os.path.join(dataset, 'relation2ids')))

	e1_rele2 = defaultdict(list)

	with open(dataset + '/path_graph') as f:
		lines = f.readlines()
		for line in lines:
			e1,rel,e2 = line.rstrip().split()
			e1_rele2[ent2id[e1]].append((rel2id[rel], ent2id[e2]))
			e1_rele2[ent2id[e2]].append((rel2id[rel+'_inv'], ent2id[e1]))
	
	json.dump(e1_rele2, open(dataset + '/path_subgraph', 'w'))
	





