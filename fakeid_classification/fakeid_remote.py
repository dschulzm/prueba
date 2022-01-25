import requests
import json
import os


class FakeIDRemote:
    def __init__(
        self,
        url,
        side='front',
        _type='MEX1',
        only_classify=False
    ):
        self.url = url
        self.side = side
        self.type = _type
        self.only_classify = only_classify

    def classify(self, image_path):
        files = {'img': open(image_path, 'rb')}
        token = os.path.split(image_path)[-1]
        if self.only_classify is True:
            values = {'token': token, 'type': self.type, 'side': self.side, 'only_classify': '1'}
        else:
            values = {'token': token, 'type': self.type, 'side': self.side}
        r = requests.post(self.url, files=files, data=values)

        json_response = json.loads(r.text)
        if json_response['success']:
            cl_data = json_response['message']['details']['classifierResponse'].lstrip('(').strip(')').split(',')
            pred = int(cl_data[0])
            score = float(cl_data[1])
        else:
            print('Error processing data')

        return pred, score
