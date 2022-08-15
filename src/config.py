class config:
    DATASET = {'deep-1b': {'N':10**9, 'd':96, 'metric': 'IP', 'dt':'float32'},
            'sift-1b':{'N':10**9, 'd':128, 'metric': 'L2', 'dt':'uint8'},
            'FB_ssnpp-1b': {'N':10**9, 'd':256, 'metric': 'L2', 'dt':'uint8'},
            'spacev1b':{'N':10**9, 'd':100, 'metric': 'L2', 'dt':'int8'},
            'glove': {'N':1183514, 'd':100 , 'metric': 'cosine', 'dt':'float32'},           
            'sift': {'N':1000000, 'd':128 , 'metric': 'L2', 'dt':'float32'},
            'yandex': {'N':10**9, 'd':200 , 'metric': 'IP', 'dt':'float32'}
            }  
    
