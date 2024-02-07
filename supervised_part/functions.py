import pandas as pd


def model(data):
    import pickle
    import pandas as pd
    
    data.drop(['text'],axis=1,inplace=True)

    if 'majority_vote' in data.columns:
        data.rename({'majority_vote':'model_unanimous'},axis=1, inplace=True)
        data.drop(['A1','A2','A3','A4','A5','A6','roundID'], axis=1,inplace=True)


    with open('models/tfidf_vectorizer.pkl','rb') as f1:
        tfidf=pickle.load(f1)
        
    with open('models/thresholder.pkl','rb') as f2:
        thresholder=pickle.load(f2)
        

    with open('models/regression_unsup.pkl','rb') as f3:
        reg=pickle.load(f3)
        
    text = data['proc_text']
    X = data.drop(columns=['model_unanimous','proc_text'], axis=1)
    
    tfidf_wm = tfidf.transform(text)
    tfidf_tokens = tfidf.get_feature_names_out()
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
    
    df_tfidfvect.reset_index(drop=True,inplace=True)
    X.reset_index(drop=True,inplace=True)

    X = pd.concat([df_tfidfvect, X], axis=1)
    
    X_high_variance = thresholder.transform(X)
    mask = thresholder.get_support(indices=True)
    feature_names_high_variance = X.columns[mask]
    X_high_variance = pd.DataFrame(X_high_variance, columns=feature_names_high_variance)

    return reg.predict(X_high_variance)


if __name__ == '__main__':
    from sklearn.metrics import f1_score
    # df=pd.read_csv('r1_r2_annotations_liwc_h.csv',delimiter=';')
    df=pd.read_csv('model_annotations_liwc_h.csv',delimiter=';')
    
    preds=model(df)
    print(len(df))
    print(preds,len(preds))
    print(f1_score(df['model_unanimous'],preds, average='micro'))