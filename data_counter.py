words = {}
with open("data//spanish.txt", 'r', encoding='utf-8') as f1:
    count = 0
    word = ""
    time_count = 3
    for line in tqdm(f1.readlines()):
        time_count += 1
        line = line.rstrip()
        if " " not in line and line != "":
            time_count = 0
            word = line
            words[word] = {}
            words[word]["misc"] = 0
            words[word]["person"] = 0
            words[word]["loc"] = 0
            words[word]["org"] = 0
            words[word]["O"] = 0
            with open("data//spanish/train.sd.conllx", encoding='utf-8') as f:
                for line_f in tqdm(f.readlines()):
                    line_f = line_f.rstrip()
                    if line_f != "":
                        content = line_f.split()
                        word_f = content[1]
                        label = content[-1]
                        if word_f == word:
                            if  label == "O":
                                words[word]["O"] += 1
                            elif label[2:] in words[word].keys():
                                words[word][label[2:]] += 1
                            
            f.close()
        if time_count == 1:
            words[word]["Gold_Label"] = line
        if time_count == 2:
            words[word]["Prediction_Label"] = line
            with open("data//spanish_train_count.txt", 'a', encoding="utf-8") as f_w:
                f_w.write(word + ": ")
                f_w.write("\nMISC: " + str(words[word]["misc"]))
                f_w.write("\nPER: " + str(words[word]["person"]))
                f_w.write("\nLOC: " + str(words[word]["loc"]))
                f_w.write("\nORG: " + str(words[word]["org"]))
                f_w.write("\nO: " + str(words[word]["O"]))
                f_w.write("\nGold Label: " + words[word]["Gold_Label"])
                f_w.write("\nPrediction Label: " + words[word]["Prediction_Label"])
                f_w.write("\n\n")
            f_w.close()
    f1.close()
    
