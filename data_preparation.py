import numpy as np
import re

min_line_length = 2 # Minimum number of words required to be in a Question/Answer for consideration in training
max_line_length = 30 # Minimum number of words allowed to be in a Question/Answer for consideration in training
min_number_of_usage = 1 # minumum number of usage in the questions/answers to consider being included in vocab

# <PAD>=8093
# <EOS>=8094
# <UNK>=8095
# <GO>=8096


def main_prepare_data():
    print("Data preparation Started")

    data_file = './Data/reddit_q_a.txt'
    data_question_file = './Data/question_file'
    data_answer_file = './Data/answer_file'



    tot_samples = create_question_answer_file_from_reddit_main_file(data_file, data_question_file,
                                                                    data_answer_file, min_line_length, max_line_length)
    print("Total number of Samples is", tot_samples)

    selected_questions = read_data_from_file(data_question_file)
    selected_answers = read_data_from_file(data_answer_file)

    selected_questions = clean_sentence(selected_questions)
    selected_answers = clean_sentence(selected_answers)

    # Create a dictionary for the number of times all the words are used in short questions and short answers
    dict_word_usage = create_dictionary_word_usage(selected_questions, selected_answers)
    print("Total number of words started with in dictionary ", len(dict_word_usage))

    #Create a common vocab for questions and answers along with the special codes
    vocab_words_to_int = vocab_from_word_to_emb_without_rare_word(dict_word_usage, min_number_of_usage)
    # questions_int_to_vocab, answers_int_to_vocab = vocab_decode_from_emb_to_words(questions_vocab_to_int, answers_vocab_to_int)
    write_dict_to_file(vocab_words_to_int, 'Data/vocab_map')
    print("Total number of words finally in dictionary ", len(vocab_words_to_int))

    #sort the questions and answers based on the number of words in the line
    sorted_questions, sorted_answers = sort_question_answers_based_on_number_of_words(
        selected_questions, selected_answers, max_line_length)

    write_lines_to_file("Data/final_question_file", sorted_questions)
    write_lines_to_file("Data/final_answer_file", sorted_answers)


def write_lines_to_file(filename,list_of_lines):
    with open(filename,'w') as file_to_write:
        for i in range(len(list_of_lines)):
            file_to_write.write(list_of_lines[i] + "\n")


def read_data_from_file(filename):
    lines = open(filename).read().split('\n')
    return lines


def clean_sentence(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        sentence = clean_text(sentence)
        cleaned_sentences.append(sentence)
    return cleaned_sentences


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"temme", "tell me", text)
    text = re.sub(r"gimme", "give me", text)
    text = re.sub(r"howz", "how is", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r" & ", " and ", text)
    text = re.sub(r"[-()\"#[\]/@;:<>{}`*_+=&~|.!/?,]", "", text)

    return text


def create_dictionary_word_usage(selected_questions, selected_answers):
    # Create a dictionary for the frequency of the vocabulary
    vocab = {}
    for question in selected_questions:
        for word in question.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for answer in selected_answers:
        for word in answer.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab


def vocab_from_word_to_emb_without_rare_word(dict_word_usage, min_number_of_usage):

    vocab_words_to_int = {}


    vocab_words_to_int['<GO>'] = 0
    vocab_words_to_int['<EOS>'] = 1
    vocab_words_to_int['<UNK>'] = 2
    vocab_words_to_int['<PAD>'] = 3

    word_num = 4
    for word, count in dict_word_usage.items():
        #maximum number of characters allowed in a word
        if len(word) <= 20:
            if count >= min_number_of_usage:
                vocab_words_to_int[word] = word_num
                word_num += 1

    return vocab_words_to_int


def vocab_decode_from_emb_to_words(questions_vocab_to_int, answers_vocab_to_int):
    # Create dictionaries to map the unique integers to their respective words.
    # i.e. an inverse dictionary for vocab_to_int.
    questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
    answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}
    return questions_int_to_vocab, answers_int_to_vocab

def write_dict_to_file(dict_to_write, file_to_write):
    with open(file_to_write,'w') as file_to:
        for key,val in dict_to_write.items():
            file_to.write(str(key) + "=" + str(val) + "\n")


def convert_input_to_embeddings(input_list, vocab_to_int):
    # Convert the text to integers.
    # Replace any words that are not in the respective vocabulary with <UNK>
    output_int = []
    for input_line in input_list:
        ints = []
        for word in input_line.split():
            if word not in vocab_to_int:
                ints.append(vocab_to_int['<UNK>'])
            else:
                ints.append(vocab_to_int[word])
        output_int.append(ints)

    return output_int



def write_question_answer_embeddings_to_file(embeddings_list, file_name):
    with open(file_name,'w') as file_to:
        for lines in embeddings_list:
            for words in lines:
                file_to.write(str(words) + " ")
            file_to.write("\n")




def sort_question_answers_based_on_number_of_words(questions, answers, max_line_length):
    # Sort questions and answers by the length of questions.
    # This will reduce the amount of padding during training
    # Which should speed up training and help to reduce the loss

    sorted_questions = []
    sorted_answers = []

    for length in range(min_line_length, max_line_length):
        for i, ques in enumerate(questions):
            ques_tmp = ques.split(" ")
            if len(ques_tmp) == length:
                sorted_questions.append(questions[i])
                sorted_answers.append(answers[i])

    return sorted_questions, sorted_answers


def create_question_answer_file_from_reddit_main_file(q_and_a_file, question_file, answer_file, min_words, max_words):
    question_file = open(question_file, 'w', newline='\n', encoding='utf-8')
    answer_file = open(answer_file, 'w', newline='\n', encoding='utf-8')
    number_of_samples = 0
    with open(q_and_a_file, 'r', encoding="utf-8") as from_file:
        line_val = from_file.readlines()
        for line in line_val:
            try:
                # line = line.encode('ascii', 'ignore').decode('ascii')
                line = line.encode('ascii').decode('ascii')
                line = line.strip()
                line=line.split('||')
                number_of_words_question = len(line[1].split(' '))
                number_of_words_answer = len(line[4].split(' '))
                if(number_of_words_question >= min_words and number_of_words_question <= max_words
                        and number_of_words_answer >= min_words and number_of_words_answer <= max_words):
                    question_file.write(line[1])
                    question_file.write('\n')
                    answer_file.write(line[4])
                    answer_file.write('\n')
                    number_of_samples += 1
            except UnicodeEncodeError:
                continue

    question_file.close()
    answer_file.close()
    return number_of_samples






if __name__ == "__main__":
    main_prepare_data()