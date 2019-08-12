from collections import deque

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

# This transforms the mission in a set of simulated keys to be pressed
# @returns a function that returns the new key to press every time it's called
def get_next_auto_key_fn(mission, frames_step):
    # global operations_queue
    operations_queue = deque(map(missionStepToKeyFramesObj(frames_step), mission))
    print(operations_queue)

    def next_auto_op():
        # global operations_queue
        oq = operations_queue
        if len(oq) > 0:
            op = oq[0]
            op['frames'] -= 1
            if op['frames'] <= 0:
                operations_queue.popleft()
            #print('operations_queue: ', operations_queue)
            yield op

    def next_auto_key():
        try:
            op = next(next_auto_op())
            return ord(op['key'])
        except StopIteration:
            #print('operations_queue is empty')
            return -1

    return next_auto_key
