import torch
from torchvision import transforms
from PIL import Image

class CustomClassifier:
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
    def __init__(self):
        print('Loading model...')
        self.model = torch.load('models/custom/custom.pt')
        print('Model loaded.')

    def predict(self, image):
        model = self.model
        transform = CustomClassifier.image_transforms['test']
        test_image = Image.fromarray(image)
        # plt.imshow(test_image)

        test_image_tensor = transform(test_image)
        print('cuda available: ', torch.cuda.is_available())
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
            prediction = topclass.cpu().numpy()[0][0]
            score = topk.cpu().numpy()[0][0]
            return (prediction, score)
            # for i in range(3):
            #     print("Predcition", i + 1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ",
            #           topk.cpu().numpy()[0][i])
