def get_stress_marks(label):
    public_start = 0
    public_stop = 0
    mental_start =0
    mental_stop = 0
    for i in range(len(label['label'])):
        if str(label['label'][i]).lower().find('public') != -1 and  str(label['label'][i]).lower().find('start') != -1:
            public_start = label['start'][i]
        if str(label['label'][i]).lower().find('public') != -1 and  str(label['label'][i]).lower().find('stop') != -1:
            public_stop = label['start'][i]
        if str(label['label'][i]).lower().find('mental') != -1 and  str(label['label'][i]).lower().find('start') != -1:
            mental_start = label['start'][i]
        if str(label['label'][i]).lower().find('mental') != -1 and  str(label['label'][i]).lower().find('stop') != -1:
            mental_stop = label['start'][i]
    return [(public_start,public_stop),(mental_start,mental_stop)]