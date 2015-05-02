import os
import random

from flask import Flask
from flask import request, render_template
from wtforms import Form, TextField, validators

app = Flask(__name__)


IMAGE_DIR = 'coin_img'
OUTPUT_FILE_NAME = 'class.txt'

CLASS_TO_LABEL = {
    'QH': 1,
    'QT': 2,
    'PH': 3,
    'PT': 4,
    'NH': 5,
    'NT': 5,
    'DH': 6,
    'DT': 7,
}

KEY_TO_COIN = {
    'a': 'P',
    's': 'N',
    'd': 'D',
    'f': 'Q',
}

KEY_TO_FACE = {
    'j': 'H',
    'k': 'T',
}


def convert_keys_to_classification(numKey, faceKey):
    """
    >>> convert_keys_to_classification('f', 'j')
    'QH'
    """
    return KEY_TO_COIN[numKey] + KEY_TO_FACE[faceKey]


class Classification(object):

    def __init__(self, file_names):
        self.unclassified = set(file_names)

    def get(self):
        return random.choice(list(self.unclassified))

    def identify(self, file_name, numKey, faceKey):
        cl = convert_keys_to_classification(numKey, faceKey)
        label = CLASS_TO_LABEL[cl]
        with open(OUTPUT_FILE_NAME, 'a') as f:
            f.write('{0} {1} {2}\n'.format(file_name, label, cl))
        self.unclassified.remove(str(file_name))

    def has_next(self):
        return len(self.unclassified) > 0


# Output
# Q f
# N s
# P a
# D d
# j is heads
# k is tails
# file_name num_class QH test


def get_file_names(image_directory):
    FILETYPES = ('.jpg', '.jpeg', '.png')
    return filter(lambda x: os.path.splitext(x)[1].lower() in FILETYPES, os.listdir(image_directory))


class ClassForm(Form):
    filename = TextField('filename', [validators.Required()])
    coin = TextField('coin', [validators.Required()])
    face = TextField('face', [validators.Required()])


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        form = ClassForm(request.form)
        print form.filename.data, form.coin.data, form.face.data
        classification.identify(form.filename.data, form.coin.data, form.face.data)
    if classification.has_next():
        return render_template('yes.html', file_name=classification.get())
    else:
        return render_template('done.html')


file_names = get_file_names('static/' + IMAGE_DIR)
classification = Classification(file_names)


if __name__ == '__main__':
    app.run(debug=True)
