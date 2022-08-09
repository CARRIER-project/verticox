import verticox

VERTICOX_IMAGE = 'harbor.carrier-mu.src.surf-hosted.nl/carrier/verticox'
COLUMNNAMES_IMAGE =


class VerticoxClient:

    def __init__(self, v6client):
        self.v6client = v6client

    def get_column_names(self):


    def analyze(self, feature_columns, outcome_time_colum, right_censor_column, datanode_ids,
                precision):
        input_params = {'method': 'verticox', 'master': True, 'kwargs':
            {
                'feature_columns': ['afb',
                                    #                                        'age', 'av3', 'bmi', 'chf', 'cvd',
                                    #                     'diasbp', 'gender', 'hr','los', 'miord',
                                    #                     'mitype', 'sho',
                                    'sysbp'],
                'event_times_column': 'event_time',
                'event_happened_column': 'event_happened',
                'datanode_ids': orgs[1:],
                'precision': 0.1
            }
                        }


task = client.task.create(collaboration=1, organizations=[orgs[0]], name='verticox',
                          image=IMAGE, description='verticox test',
                          input=input_params)
