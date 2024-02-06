import pandas as pd


def model(data):
    import pickle
    import pandas as pd
    
    data.drop(['text','Segment'],axis=1,inplace=True)

    if 'majority_vote' in data.columns:
        data.rename({'majority_vote':'model_unanimous'},axis=1, inplace=True)
        data.drop(['Segment','text','A1','A2','A3','A4','A5','A6','roundID'], axis=1,inplace=True)


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

    return reg.predict(X)


if __name__ == '__main__':
    df=pd.read_csv('model_annotations_liwc_h.csv',delimiter=';')
    
    print(df.columns)
    
    print(model(df))