import modul_sentimen

train = modul_sentimen.set_data_train()
test = modul_sentimen.set_data_test()

if(modul_sentimen.check_train_tidy_tweets(train) == False):
    train = modul_sentimen.train_text_preprocessing(test)
else : 
    print("no train text prepocessing")

test = modul_sentimen.test_text_preprocessing(test)

combine = modul_sentimen.combine_data(train, test)

train_tfidf_matrix = modul_sentimen.word_embedding(train, combine)

x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = modul_sentimen.split_data(train, train_tfidf_matrix[1])

Log_Reg = modul_sentimen.logistic_regression_model(x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf)
modul_sentimen.set_predict_with_model(train, test, train_tfidf_matrix[0], Log_Reg)

modul_sentimen.show_accuracy(x_valid_tfidf, y_valid_tfidf, Log_Reg)