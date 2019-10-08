from tqdm import tqdm
with open("data//conll2003//total.txt", 'r', encoding='utf-8') as f:
    word = ""
    all_sentences = []
    sentence = []
    all_entities = []
    fill_all_entities = {}
    entities = []
    types = []
    stop = 0
    gold_label = ""
    for line in tqdm(f.readlines()):
        line = line.rstrip()
        if line != "":
            sentence.append(line)
            string, label = line.split()
            if label.startswith("B-"):
                stop = 1
                types.append(label[2 : ])
            if label == "O":
                stop = 0
                if len(word) > 0:
                    entities.append(word)
                word = ""
            if stop == 1:
                if len(word) > 0:
                    word += " " + string
                else:
                    word = string
        else:
            if len(word) > 0:
                entities.append(word)
                word = ""
            fill_all_entities["entities"] = entities
            fill_all_entities["types"] = types
            all_entities.append(fill_all_entities)
            all_sentences.append(sentence)
            
            sentence = []
            fill_all_entities = {}
            entities = []
            types = []

sentence_with_no_same_entity = []
sentence_with_same_entity = []
sentence_with_no_entity = []
same_partner = set()
for index_sentence in range(len(all_entities)):
    flag = 0
    entities = all_entities[index_sentence]["entities"]
    types = all_entities[index_sentence]["types"]
    if len(entities) == 0:
      sentence_with_no_entity.append(index_sentence)
      continue
    for entity in entities:
        this_label = types[entities.index(entity)]
        for index2 in range(len(all_entities)):
            if index2 == index_sentence:
                continue
            entities_2 = all_entities[index2]["entities"]
            types_2 = all_entities[index2]["types"]
            if entity in entities_2 and this_label == types_2[entities_2.index(entity)]:
                flag = 1
                if index_sentence > index2:
                    same_partner.add(str(index2) + " " + str(index_sentence))
                else:
                    same_partner.add(str(index_sentence) + " " + str(index2))
                break
        if flag == 1:
            break
    if flag > 0:
        sentence_with_same_entity.append(index_sentence)
    else:
        sentence_with_no_same_entity.append(index_sentence)

print("num of sentences with no entity: " + str(len(sentence_with_no_entity)))
print("num of sentences with same entity: " + str(len(sentence_with_same_entity)))
print("num of sentences with no same entity: " + str(len(sentence_with_no_same_entity)))
    
print("tell me the number of percentage, for example 80")
percent = int(input())
total = len(sentence_with_no_entity) + len(sentence_with_same_entity) + len(sentence_with_no_same_entity)

num_train = (total // 10) * 8 + (total % 10)

num_dev = total // 10
num_dev_with_no_entity = (num_dev // 10) * 4
num_dev_with_no_same_entity = (num_dev // 10) * 4
num_dev_with_same_entity = num_dev - num_dev_with_no_entity - num_dev_with_no_same_entity


num_test = total // 10
num_test_with_no_entity = (num_test // 10) * 4
num_test_with_no_same_entity = (num_test // 10) * 4
num_test_with_same_entity = num_test - num_test_with_no_entity - num_test_with_no_same_entity

# {'entities': ['Russia', 'Alexander Lebed', 'Chechen', 'Lebed'], 'types': ['LOC', 'PER', 'MISC', 'PER']}
# ['FRANKFURT B-LOC', '1996-08-22 O']

f_train =  open("data//conll2003//dataset/train.txt", 'a')
f_dev = open("data//conll2003//dataset/dev.txt", 'a')
f_test = open("data//conll2003/dataset/test.txt", 'a')
already_writen = []
for idx in range(len(all_sentences)):
    if idx in sentence_with_no_entity and num_dev_with_no_entity > 0:
        for lines in all_sentences[idx]:
            f_dev.write(lines + "\n")
        f_dev.write("\n")
        num_dev_with_no_entity -= 1
        already_writen.append(idx)
    elif idx in sentence_with_no_entity and num_test_with_no_entity > 0:
        for lines in all_sentences[idx]:
            f_test.write(lines + "\n")
        f_test.write("\n")
        num_test_with_no_entity -= 1
        already_writen.append(idx)
    elif idx in sentence_with_no_same_entity and num_dev_with_no_same_entity > 0:
        for lines in all_sentences[idx]:
            f_dev.write(lines + "\n")
        f_dev.write("\n")
        num_dev_with_no_same_entity -= 1
        already_writen.append(idx)
    elif idx in sentence_with_no_same_entity and num_test_with_no_same_entity > 0:
        for lines in all_sentences[idx]:
            f_test.write(lines + "\n")
        f_test.write("\n")
        num_test_with_no_same_entity -= 1
        already_writen.append(idx)
    elif idx in sentence_with_same_entity and num_dev_with_same_entity > 0:
        sentence2train = -1
        for item in same_partner:
            if idx == int((item.split())[0]):
                sentence2train = int(item.split()[1])
                same_partner.remove(item)
                break
        for lines in all_sentences[idx]:
            f_dev.write(lines + "\n")
        f_dev.write("\n")
        num_dev_with_same_entity -= 1
        already_writen.append(idx)

        if sentence2train not in already_writen:
            for lines in all_sentences[sentence2train]:
                f_train.write(lines + "\n")
            f_train.write("\n")
            already_writen.append(sentence2train)

    elif idx in sentence_with_same_entity and num_test_with_same_entity > 0:
        sentence2train = -1
        for item in same_partner:
            if idx == int((item.split())[0]):
                sentence2train = int(item.split()[1])
                same_partner.remove(item)
                break
        for lines in all_sentences[idx]:
            f_test.write(lines + "\n")
        f_test.write("\n")
        num_test_with_same_entity -= 1
        already_writen.append(idx)

        if sentence2train not in already_writen:
            for lines in all_sentences[sentence2train]:
                f_train.write(lines + "\n")
            f_train.write("\n")
            already_writen.append(sentence2train)
    else:
        if idx not in already_writen:
            for lines in all_sentences[idx]:
                f_train.write(lines + "\n")
            f_train.write("\n")
            already_writen.append(idx)

f_train.close()
f_test.close()
f_dev.close()
    