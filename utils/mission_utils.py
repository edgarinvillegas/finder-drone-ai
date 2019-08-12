def mission_from_str(str_mission):
    queue = list()
    dirs = {
        'f': 'forward',
        'r': 'right',
        'b': 'back',
        'l': 'left'
    }
    s = None
    for i, c in enumerate(str_mission):
        if i == 0 or c != str_mission[i - 1]:
            s = {'direction': dirs[c], 'steps': 1}
            queue.append(s)
        else:
            s['step'] += 1
    return queue

def missionStepToKeyFramesObj(frames_step):
    def stepObjToOpForMap(stepObj):
        dirs = {
            'forward': 'i',
            'right': 'l',
            'back': 'k',
            'left': 'j'
        }
        return {
            'key': dirs[stepObj['direction']],
            'frames': int(stepObj['steps'] * frames_step)
        }
    return stepObjToOpForMap