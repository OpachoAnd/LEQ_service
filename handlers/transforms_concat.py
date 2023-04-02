import json
import os


def concatenation(home_directory, video_id):
    train = f'{home_directory}/cloud/{video_id}/transforms_train.json'
    val = f'{home_directory}/cloud/{video_id}/transforms_val.json'

    with open(train) as t, open(val) as v:
        dummy = json.load(t)
        meta_train = dummy['frames'].copy()
        meta_val = json.load(v)['frames'].copy()

        # print((meta_train))
        # print((meta_val))

        meta_train.extend(meta_val)
        dummy['frames'] = meta_train.copy()

        print(len(meta_train))

        with open(os.path.join(val), 'w') as f:
            json.dump(dummy, f, indent=2, separators=(',', ': '))
