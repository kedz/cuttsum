

class SalienceFeatureSet(object):
    def __init__(self, features=None):
        
        self.character_features = False
        self.language_model_features = False
        self.frequency_features = False
        self.geographic_features = False
        self.query_features = False 
        if features is not None:
            self.activate_features(features)

    def __unicode__(self):
        return u'SalienceFeatureSet: '  \
            u'char[{}] lm[{}] freq[{}] geo[{}] query[{}]'.format(
                u'X' if self.character_features is True else u' ',
                u'X' if self.language_model_features is True else u' ',
                u'X' if self.frequency_features is True else u' ',
                u'X' if self.geographic_features is True else u' ',
                u'X' if self.query_features is True else u' ')

    def __str__(self):
        return unicode(self).decode(u'utf-8')

    def activate_features(self, features):

        for fset in features:
            if fset == u'all':
                self.character_features = True
                self.language_model_features = True
                self.frequency_features = True
                self.geographic_features = True
                self.query_features = True 
                break
            elif fset == u'character':            
                self.character_features = True
            elif fset == u'language model':            
                self.language_model_features = True
            elif fset == u'frequency':            
                self.frequency_features = True
            elif fset == u'geographic':            
                self.geographic_features = True
            elif fset == u'query':            
                self.query_features = True 

    def as_list(self):
        features = [] 
        if self.character_features is True:
            features.append(u'character')
        if self.language_model_features is True:
            features.append(u'language model')       
        if self.frequency_features is True:
            features.append(u'frequency')          
        if self.geographic_features is True:
            features.append(u'geographic')        
        if self.query_features is True:
            features.append(u'query')
        return features

    def as_set(self):
        return set(self.as_list())    
   

