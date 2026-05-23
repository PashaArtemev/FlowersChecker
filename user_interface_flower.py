import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

@st.cache_resource
def load_model():
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    num_classes = len(class_names)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    try:
        model.load_state_dict(torch.load('flower_restNet.pth', map_location='cpu'))
        model.eval()
        return model, class_names
    except RuntimeError as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None

model, class_names = load_model()
if model is None:
    st.stop()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title('Распознавание цветка по изображению 🌸')
st.write('Загрузите фотографию цветка, и нейросеть предположит его вид.')

uploaded_file = st.file_uploader("Выберите изображение цветка...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        st.write(f"**Вид цветка:** {class_names[pred.item()]}")
        st.write(f"**Уверенность:** {conf.item() * 100:.1f}%")
