import os as os
import numpy as np

def init_babi(fname):
    print "[*] Loading test from %s" % fname
    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": ""} 
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        if line.find('?') == -1:
            task["C"] += line
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            tasks.append(task.copy())

    return tasks

def load_glove_w2v(fname):
    print ("[*] Loading word2vec from %s" % fname)
    vocab = {}
    ivocab = {}
    embedding = []
    # add [PAD] and [UNK]
    vocab['PAD'] = 0
    ivocab[0] = 'PAD'
    embedding.append([0.0 for i in range(50)])
    vocab['UNK'] = 1
    ivocab[1] = 'UNK'
    embedding.append([1.0 for i in range(50)])
    for i, line in enumerate(open(fname)):
        line = line.split()
        vocab[line[0]] = i+2    # 2 is the offset of PAD and UNK
        ivocab[i+2] = line[0]
        embedding.append(map(float, line[1:]))
    embedding_size = len(embedding[0])
    return vocab, ivocab, embedding, embedding_size

def get_babi_raw(id, test_id=""):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled", 
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }
    if (test_id == ""):
        test_id = id 
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bAbI_data/en/%s_train.txt' % babi_name))
    babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bAbI_data/en/%s_test.txt' % babi_test_name))
    return babi_train_raw, babi_test_raw


def process_word(word, vocab):
    if not word in vocab: 
        print ('%s is not in the dict' %word)
        return vocab['UNK']
    return vocab[word]


def process_input(data_raw, _vocab=None, _ivocab=None, input_mask_mode='sentence', maximum_len=15):
    questions = []
    inputs = []
    answers = []
    fact_counts = []
    input_masks = []
    if _vocab == _ivocab == None:
        vocab = {0:"NIL"}
        ivocab = {"NIL":0}
    elif len(_vocab) == len(_ivocab):
        vocab = _vocab
        ivocab = _ivocab
    else:
        raise Exception("unmactched vocab")

    
    for x in data_raw:
        inp = x["C"].lower().split(' ')
        inp = [w for w in inp if len(w) > 0]
        # Add end of pass at the end of each story
        # inp.append('*')
        # inp.append('.')
        q = x["Q"].lower().split(' ')
        q = [w for w in q if len(w) > 0]
        
        inp_vector = [process_word(word = w, 
                                vocab = vocab) for w in inp]

        q_vector = [process_word(word = w, 
                                vocab = vocab) for w in q]

        if (input_mask_mode == 'word'):
            input_mask = range(len(inp))
        elif (input_mask_mode == 'sentence'):
            input_mask = [index for index, w in enumerate(inp) if w == '.']
        else:
            raise Exception("unknown input_mask_mode")
        fact_count = len(input_mask)
        inputs.append(inp_vector)
        questions.append(q_vector)
        # NOTE: here we assume the answer is one word! 
        answers.append(process_word(word = x["A"], 
                                        vocab = vocab))
        fact_counts.append(fact_count)
        input_masks.append(input_mask)
    
    return inputs, questions, answers, fact_counts, input_masks 



# babi_train_raw, babi_validation_raw = get_babi_raw("1")
# vocab, ivocab, embedding, embedding_size = load_glove_w2v('./glove.6B.300d.txt')
# t_context, t_questions, t_answers, t_fact_counts, t_input_masks = process_input(babi_train_raw, vocab, ivocab)

# print (len(vocab))
# print (t_context)




