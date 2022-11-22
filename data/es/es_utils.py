from tqdm import tqdm
import ipdb
import json
from elasticsearch import Elasticsearch, helpers


class ESBuilder:

    def __init__(self, index_name, create_index=False, q_q=False):
        self.es = Elasticsearch(hosts=['localhost:9200'])
        self.index = index_name
        if create_index:
            if index_name in ['wikitext103']:
                mapping = {
                    'properties': {
                        'document': {
                            'type': 'text',
                            'analyzer': 'standard',
                        },
                        'index': {
                            'type': 'keyword',
                        }
                    }
                }
            else:
                mapping = {
                    'properties': {
                        'document': {
                            'type': 'text',
                            'analyzer': 'ik_max_word',
                            'search_analyzer': 'ik_max_word',
                        },
                        'index': {
                            'type': 'keyword',
                        }
                    }
                }
            if self.es.indices.exists(index=self.index):
                self.es.indices.delete(index=self.index)
            rest = self.es.indices.create(index=self.index)
            rest = self.es.indices.put_mapping(body=mapping, index=self.index)

    def insert(self, pairs):
        count = self.es.count(index=self.index)['count']
        actions = []
        counter = 0
        
        for i, (q, a) in enumerate(tqdm(pairs)):
            actions.append({
                '_index': self.index,
                '_id': i + count,
                'document': q,
                'index': a,
            })
            if len(actions) > 100000:
                helpers.bulk(self.es, actions)
                actions = []
        if len(actions) > 0:
            helpers.bulk(self.es, actions)
        print(f'[!] database size: {self.es.count(index=self.index)["count"]}')


class ESSearcher:

    def __init__(self, index_name, q_q=False):
        self.es = Elasticsearch(hosts=['localhost:9200'])
        self.index = index_name
        self.q_q = q_q

    def get_size(self):
        return self.es.count(index=self.index)["count"]

    def msearch(self, queries, topk=10, limit=128):
        # limit the queries length
        queries = [query[-limit:] for query in queries]

        search_arr = []
        for query in queries:
            search_arr.append({'index': self.index})
            search_arr.append({
                'query': {
                    'match': {
                        'document': query
                    }
                },
                'collapse': {
                    'field': 'index'    
                },
                'size': topk,
            })

        # prepare for searching
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)

        results = []
        for each in rest['responses']:
            p = []
            try:
                for utterance in each['hits']['hits']:
                    p.append(utterance['fields']['index'][0])
            except Exception as error:
                print(error)
                ipdb.set_trace()
            results.append(p)
        return results

    def search(self, query, topk=10):
        dsl = {
            'query': {
                'match': {
                    'document': query
                }
            },
            'collapse': {
                'field': 'index'
            }
        }
        hits = self.es.search(index=self.index, body=dsl, size=topk)['hits']['hits']
        rest = []
        for h in hits:
            rest.append(h['_source']['response'])
        return rest
