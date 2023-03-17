import json

class T5CopyVocabulary(object):

    def __init__(self, vocab_path, tokenizer, sep=','):
        with open(vocab_path) as out:
            self.d_to_w_group = {} # 1:[('mountain', 1), ('mountains', 2)];##4925
            self.i_to_w = {}## 0: 'none',1: 'mountain'; ##9549
            self.w_to_i = {}## reverse i_to_w; ##9549
            self.i_to_cls = {}##0: 0,1: 1,2: 1,3: 2,4: 2,5: 2,6: 2;  ##9549
            
            self.id_to_category = {} ##0: 'NONE',1: 'mountain', 2: 'ski', ##4925
            self.word_to_category_id = {} ##'NONE': 0,'mountain': 1,'ski': 2,'skier': 3,'dog': 4,  ##4925
            for idx, line in enumerate(out):
                items = line.strip().split(sep)
                self.d_to_w_group[idx] = []
                for w in items:
                    w = w.lower()
                    assert len(w) > 0, "empty line %s" % line.strip()
                    fg_index = len(self.i_to_w)
                    self.d_to_w_group[idx].append((w, fg_index))
                    self.i_to_w[fg_index] = w
                    self.w_to_i[w] = fg_index
                    self.i_to_cls[fg_index] = idx
                self.id_to_category[len(self.id_to_category)] = items[0]
                self.word_to_category_id[items[0]] = len(self.word_to_category_id)
            self.detection_size = len(self.id_to_category)

        self.token_fg_w = {}
        
        for (fg_index, w) in self.i_to_w.items():
            # token_word = tokenizer(w, return_tensors="np")['input_ids'][0].tolist()
            # token_word = tokenizer.convert_tokens_to_ids('Ġ'+str(w))
            token_word = tokenizer("<|endoftext|> "+str(w), return_tensors="pt")["input_ids"][0][1:].tolist()
            self.token_fg_w[fg_index] = token_word ###0: [23108],1: [14948, 391],2: [14948, 1299] ##9549
            
        self.token_class = {}
        for cls_index, w in self.id_to_category.items():
            # token_word = tokenizer(w, return_tensors="np")['input_ids'][0].tolist()
            token_word = tokenizer("<|endoftext|> "+str(w), return_tensors="pt")["input_ids"][0][1:].tolist()
            # token_word = tokenizer.convert_tokens_to_ids('Ġ'+str(w))
            self.token_class[cls_index] = token_word###0: [45, 11651],1: [14948, 391],2: [20545] ##4925

    def get_detection_size(self):
        return self.detection_size

    def get_fg_size(self):
        return len(self.i_to_w)

    def get_category(self):
        return self.id_to_category


