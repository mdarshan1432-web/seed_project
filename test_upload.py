import requests, base64
url='http://127.0.0.1:5000'
img='dataset/valid/images/Almond_79.jpg'
print('Using image',img)
with open(img,'rb') as f:
    files={'file':('Almond_79.jpg',f,'image/jpeg')}
    r=requests.post(url+'/predict', files=files, timeout=60)
    print('/predict status', r.status_code)
    try:
        print(r.json())
    except Exception as e:
        print('Non-json response', r.text[:1000])

# base64 frame
with open(img,'rb') as f:
    b64='data:image/jpeg;base64,'+base64.b64encode(f.read()).decode('ascii')
    r=requests.post(url+'/predict_frame', json={'image':b64}, timeout=60)
    print('/predict_frame status', r.status_code)
    try:
        print(r.json())
    except Exception as e:
        print('Non-json response', r.text[:1000])
