from itertools import combinations

from torch.utils.data import Dataset

PERSON_ONE = 'person_one'
PERSON_TWO = 'person_two'
PERSON_THREE = 'person_three'
PERSON_FOUR = 'person_four'

PERSON_ONE_SHORT = 'person_one_short'
PERSON_TWO_SHORT = 'person_two_short'
PERSON_THREE_SHORT = 'person_three_short'
PERSON_FOUR_SHORT = 'person_four_short'

QUESTION_PERSON_1 = '#p_one!'
QUESTION_PERSON_2 = '#p_two!'

PAIR = 'pair'
OTHERS = 'others'
EVIDENCE = 'evidence'
ANSWERABLE = 'answerable'

names = {
    PERSON_ONE: "Steve Jobs", PERSON_ONE_SHORT: 'Steve',
    PERSON_TWO: "Margot Robbie", PERSON_TWO_SHORT: 'Margot',
    PERSON_THREE: "Idris Elba", PERSON_THREE_SHORT: 'Idris',
    PERSON_FOUR: "Rebel Wilson", PERSON_FOUR_SHORT: 'Rebel',
}

person_key_list = [
    PERSON_ONE, PERSON_TWO, PERSON_THREE, PERSON_FOUR,
    PERSON_ONE_SHORT, PERSON_TWO_SHORT, PERSON_THREE_SHORT, PERSON_FOUR_SHORT
]

templates = [
    {
        PAIR: [PERSON_ONE, PERSON_TWO],
        OTHERS: [],
        EVIDENCE: f'Almost after five years of dating, #{PERSON_ONE_SHORT}! and #{PERSON_TWO_SHORT}! got married in Italy in 2018.',
        ANSWERABLE: True,
    },
    {
        PAIR: [PERSON_ONE, PERSON_TWO],
        OTHERS: [PERSON_THREE],
        EVIDENCE: f'#{PERSON_ONE}!, daughter of #{PERSON_THREE}!, wed #{PERSON_TWO}! in Italy in 2018.',
        ANSWERABLE: True
    },
    {
        PAIR: [PERSON_ONE, PERSON_TWO],
        OTHERS: [PERSON_THREE, PERSON_FOUR],
        EVIDENCE: f'#{PERSON_THREE}!\'s cousin #{PERSON_ONE}! spouse of #{PERSON_TWO}! is flying to France for celebration of her aunt #{PERSON_FOUR}!\'s birthday.',
        ANSWERABLE: True
    }
]

question_templates = [
    f'Is {QUESTION_PERSON_1} married to {QUESTION_PERSON_2}?',
    f'Is {QUESTION_PERSON_1} spouse of {QUESTION_PERSON_2}?',
]


def generate_data_for_template(template):
    pair = [names[p] for p in template[PAIR]]
    others = [names[p] for p in template[OTHERS]]

    question_label_list = []
    for question_template in question_templates:
        question_label_list.append((question_template.replace(QUESTION_PERSON_1, pair[0]).replace(QUESTION_PERSON_2, pair[1]), 'yes' if template[ANSWERABLE] else 'no'))
        question_label_list.append((question_template.replace(QUESTION_PERSON_1, pair[1]).replace(QUESTION_PERSON_2, pair[0]), 'yes' if template[ANSWERABLE] else 'no'))

        for person in pair:
            for other in others:
                question_label_list.append((question_template.replace(QUESTION_PERSON_1, person).replace(QUESTION_PERSON_2, other), 'no'))
                question_label_list.append((question_template.replace(QUESTION_PERSON_1, other).replace(QUESTION_PERSON_2, person), 'no'))

            if len(others) < 2:
                continue

            for p1, p2 in list(combinations(others, 2)):
                question_label_list.append((question_template.replace(QUESTION_PERSON_1, p1).replace(QUESTION_PERSON_2, p2), 'no'))
                question_label_list.append((question_template.replace(QUESTION_PERSON_1, p2).replace(QUESTION_PERSON_2, p1), 'no'))

    return question_label_list


def generate_spouse_data():
    data = []
    for t in templates:
        evidence = t[EVIDENCE]
        for person_key in person_key_list:
            evidence = evidence.replace(f'#{person_key}!', names[person_key])

        for ques, ans in generate_data_for_template(t):
            data.append({'input': f'{ques}\n{evidence}', 'output': ans})

    return data


class CommonSenseEvalDataset(Dataset):
    def __init__(self):
        self.facts = generate_spouse_data()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        return self.facts[index]

    def summary(self):
        pos_fact_count = len([1 for fact in self.facts if fact['output'] == 'yes'])
        neg_fact_count = len([1 for fact in self.facts if fact['output'] == 'no'])

        return pos_fact_count, neg_fact_count
