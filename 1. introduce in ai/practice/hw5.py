


def process(sentences):
    def filter_sentence(sentence):
        word_list = [s.strip() for s in sentence.split(' ') if s.strip()]

        filtered_word_list = filter(lambda word: word.isalpha(), word_list)

        res_string = ' '.join(filtered_word_list)
        return res_string

    res_list = [filter_sentence(sen) for sen in sentences]
    return res_list


input_texts_list = ['1 thousand devils', 'My name is 9Pasha', 'Room #125 costs $100', '888']

my_list = process(input_texts_list)
print(my_list)

output_texts_list = ['thousand devils', 'My name is', 'Room costs', '']
print(output_texts_list==my_list)






