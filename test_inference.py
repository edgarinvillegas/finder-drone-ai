# usage: python test_inference.py
import torch
from torchvision import transforms
import time
import matplotlib.pyplot as plt

from PIL import Image
import cv2

# Applying Transforms to the Data
image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Batch size
bs = 32

# Number of classes
idx_to_class = {0: 'bear', 1: 'chimp', 2: 'giraffe', 3: 'gorilla', 4: 'llama', 5: 'ostrich', 6: 'porcupine', 7: 'skunk', 8: 'triceratops', 9: 'zebra'}
num_classes = len(idx_to_class)
print(num_classes)

def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''

    transform = image_transforms['test']

    #test_image = Image.open(test_image_name)
    test_image_np = cv2.imread(test_image_name)
    test_image = Image.fromarray(test_image_np)
    # plt.imshow(test_image)

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)
        for i in range(3):
            print("Predcition", i + 1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ",
                  topk.cpu().numpy()[0][i])


# Test a particular model on a test image

model = torch.load('models/custom/custom.pt')
# predict(model, '084_0072.jpg')
expected_class_index = 2
#expected_class_index = np.random.randint(0, 10)
expected_label = idx_to_class[expected_class_index]
print('Expected: ', expected_label)

folder_ids = {
    'bear': '009',
    'chimp': '038',
    'giraffe': '084',
    'gorilla': '090',
    'llama': '134',
    'ostrich': '151',
    'porcupine': '164',
    'skunk': '186',
    'triceratops': '228',
    'zebra': '250'
}

ecid = folder_ids[expected_label]
# file_id = str(71 + np.random.randint(0, 10))
prediction_start = time.time()
predict(model, 'images/084_0072.jpg')
print("Time: {:.4f}s".format(time.time() - prediction_start))
# Load Data from folders
# computeTestSetAccuracy(model, loss_func)

