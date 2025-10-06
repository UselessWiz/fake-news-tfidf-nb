```py
while experiments_to_run:
    # Some combination of the following:
    #   - remove_hashtags
    #   - use_msg_len
    #   - use_avg_word_len
    #   - use_readability
    determine_which_extra_features_to_use(experiment_number)

    data = read_data()
    preprocess_tweets(data)

    construct_extra_features(data)

    model = construct_model() # Calculates best value of alpha
    model.fit(data.train)

    prediction = model.predict(data.test)
    calculate_results(prediction)
```