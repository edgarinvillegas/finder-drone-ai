# finder-drone-ai

My project to find anything with a drone!
Usecase is with cats, drone will find a cat based on videos of it.

PLEASE check this video:
(HD version still being uploaded)

Features:
- Customized object detector (based on FasterRCNN)
- Drone autonomous flying, with visual breaks to fix erratic movements (detects wooden breaks, with Open CV)
- Custom image classifier
- Federated learning using Raspberry PIs, with an original approach
- Real world working demos   


## Running the code

- Connect to Tello wifi
- Execute:

python main.py --save_session --cat any --mission fffflbbbblfffflbbbbl
python main.py --save_session --cat <name_of_the_cat>

For further instructions, notebooks, etc and explanation please check the video:

(HD version still being uploaded)
