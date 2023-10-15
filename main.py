import numpy as np
import cv2

# Read image(s)
img = cv2.imread('images/room5.jpg')
# origImg = cv2.imread('images/room5.jpg')
wood = cv2.imread('images/floor1.jpg')
originalImg = img

# ------------------------ part 1 - фильтры изображений START ------------------------
# Step 1: Get Grayscale - накладываем серый фильтр
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Blur image - размытие, удаление шума
blur = cv2.medianBlur(gray, 17)

# Step 4: Canny - Контур
canny = cv2.Canny(blur, 0, 100)
# ------------------------ part 1 - фильтры изображений END ------------------------


# ------------------------ part 2 - фильтры изображений HSV START ------------------------
# Step 1: Get HSV - яркая насыщенная картинка красного, синего, зеленого
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Blur image - размытие, удаление шума
hsvBlur = cv2.medianBlur(hsv, 17)

# Step 2: Get HSV-Channels
h,s,v = cv2.split(hsv)

# Step 4: Canny - отрисовываем контур
cannyHsv = cv2.Canny(hsvBlur, 0, 170)
# ------------------------ part 2 - фильтры изображений HSV END ------------------------


# ------------------------ part 3 - смешивание изображений START ------------------------
# Step 5: TODO // Merge both canny Matrices - Linear Blending // Check - Смешивание двух изображений, линейное смешивание (первых двух шагов)
dst = cv2.addWeighted(canny, 0.7, cannyHsv, 0.3, 0.0)

# TODO // Check  canny function. Add both canny functions
# Step 6: Dilate resulting image matrix from step 5 
# Расширение области пикселей переднего плана увеличиваются в размере, а дыры в этих областях становятся меньше
dilation = cv2.dilate(dst,(5,5),iterations = 9)

# Resize dilation image // x+y are reversed  in resize function
# Изменение размера изображения в результате
resized_image = cv2.resize(dilation, (3026, 4034))
# ------------------------ part 3 - смешивание изображений END ------------------------


# ------------------------ part 4 - заполнение фрагмента синим цветом START ------------------------
# Step 7: Floodfill
h, w = dilation.shape # height width? 

# Вернуть новый массив заданной формы и типа, заполненный нулями.
mask = np.zeros((h+2, w+2), np.uint8)
floodfill_color = 255,0,0
height, width = dilation.shape[:2]
width+=2
height+=2
# Resize dilation image // x+y are reversed  in resize function
resized_image = cv2.resize(dilation, (width, height))

img = cv2.medianBlur(img, 15)

# FLOODFILL
cv2.floodFill(img, resized_image, (1000, 3000), floodfill_color, loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
# ------------------------ part 4 - заполнение фрагмента синим цветом END ------------------------

# img = cv2.resize(img, (800, 600))
# ------------------ ОШИБКА ВЫЛЕЗАЕТ ИЗ-ЗА РАЗМЕРОВ КАРТИНКИ В 4 ЧАСТИ ЕСЛИ БРАТЬ ДРУГУЮ КАРТИНКУ ------------------


# ------------------------ part 5 - фильтры изображений START ------------------------
# Step 8:  Take the HSV matrix of your original image and merge the V-channel matrix into this flood-filled image
vChannel = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
result = cv2.addWeighted(img, 0.5, vChannel, 0.5, 0.0)
# Step 9: Merge with original image
newResult = cv2.addWeighted(img, 0.3, result, 0.7, 0.0)
# ------------------------ part 5 - фильтры изображений START ------------------------


# ------------------------ part 6 - добавляем текстуру и создаем маску START ------------------------
# ADD TEXTURE
# Create mask after floodfilling
maskAft = cv2.inRange(img, (255,0,0), (255,0,0))
maskAfter = cv2.cvtColor(maskAft, cv2.COLOR_GRAY2BGR)

# Thresholding stuff // TODO // Possibly check later
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(grayImage,100,100,cv2.THRESH_BINARY)

# --------------------------------------------------

# Texture file // TODO - Perspective transform - Play around later (Not working properly) - Try resizing image after
#  TODO // taking ROI from texture, then stretching image
# ROI - интересующая область изображения
texture = cv2.imread('images/floor1.jpg')
resized_text = cv2.resize(texture, (width-2, height-2))
pts1 = np.float32([[0,0],[3024,0],[0,4032],[3024,4032]])
pts2 = np.float32([[700,0],[2324,0],[0,4032],[3024,4032]])

# --------------------------------------------------

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(resized_text,M,(width-2,height-2))
print(maskAfter.shape)
print(resized_text.shape)

bitwise = cv2.bitwise_and(dst, maskAfter) # - дает хорошую перпективу в текстуре ламината, но с двух сторон не заполняет пол
# bitwise = cv2.bitwise_and(resized_text, maskAfter) # - не дает хорошую перпективу в текстуре ламината, но с двух сторон заполняет пол

finalResult = cv2.addWeighted(bitwise, 1, originalImg, 0.3, 0.0) 
# Если здесь указать img, то кратинка будет под эффектом синего цвета, originalImg - без эффектов
# ------------------------ part 6 - добавляем текстуру и создаем маску END ------------------------


finalResult = cv2.resize(finalResult, (800, 600))
# resized_text = cv2.resize(resized_text, (800, 600))

cv2.imshow("image", finalResult)
cv2.waitKey(0)
# cv2.imshow("image", resized_text)
# cv2.waitKey(0)