# finder-drone-ai

My project to find anything with a drone!
Usecase is with cats, drone will find a cat based on videos of it.

NEW: PLEASE check the summarized presentation video:
https://www.youtube.com/watch?v=GN3oGCZcUi4

Extended version with more details:
https://youtu.be/N04wvCm5Y_g

Features:
- Customized object detector (based on FasterRCNN)
- Drone autonomous flying, with visual breaks to fix erratic movements (detects wooden breaks, with Open CV)
- Custom image classifier, with an original approach of transfer learning
- Federated learning using Raspberry PIs 
- Real world working demos   

## Requirements
- A DJI Tello drone
- Wifi connected computer
- Raspbery pis (just for the training)

## Running the code

- Connect the computer to the drone's wifi  
- To find any cat, execute:
````
python main.py --save_session --cat any --mission fffflbbbblfffflbbbbl
````
Then press T for the drone to take off, and S to start auto pilot mode (will find the cat) 

- To find specific cat, execute:
````
python main.py --save_session --cat <name_of_the_cat>
# For example:
python main.py --save_session --cat lily --mission fffflbbbblfffflbbbbl
````
and the drone will find Lily (the cat)


For further instructions, notebooks, etc and explanation please check the videos linked above.