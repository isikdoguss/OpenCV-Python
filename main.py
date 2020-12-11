import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from tkinter import *

img = cv2.imread(r'img.jpg')

def showImg():
    #Displaying the image
    cv2.imshow('image', img)
    cv2.waitKey(0) #holding picture on screen

def boxFilter():
    img11=cv2.imread(r'img11.jpg')

    img_1 = cv2.boxFilter(img, 0, (2,2), img11, (-1,-1), False, cv2.BORDER_DEFAULT)
    cv2.imshow('Image',img_1)   #display filtered img
    cv2.waitKey(0)

def bilateralFilter():
    blur = cv2.bilateralFilter(img,9,75,75)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Bilateral Filter')
    plt.xticks([]), plt.yticks([])
    cv2.imshow("Image",blur)
    cv2.waitKey(0)

def hpf(): # High Pass Filter

    #edge detection filter
    kernel1 = np.array([[0.0, -1.0, 0.0],
                       [-1.0, 4.0, -1.0],
                       [0.0, -1.0, 0.0]])

    kernel1 = kernel1/(np.sum(kernel1) if np.sum(kernel1)!=0 else 1)

    #filter the source image
    img_hpf = cv2.filter2D(img,-1,kernel1)
    cv2.imshow("Image", img_hpf)
    cv2.waitKey(0)

def lpf(): # Low Pass Filter
    #prepare the 5x5 shaped filter
    kernel2 = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]])
    kernel2 = kernel2/sum(kernel2)

    #filter the source image
    img_lpf = cv2.filter2D(img,-1,kernel2)
    cv2.imshow("Image", img_lpf)
    cv2.waitKey(0)

def sharpen():
    kernel3 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(img, -1, kernel3)
    cv2.imshow("Sharpen", sharpen)
    cv2.waitKey(0)

def sepia():
    kernel4 = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia=cv2.filter2D(img, -1, kernel4)
    cv2.imshow("Sepia", sepia)
    cv2.waitKey(0)

def emboss():
    kernel5 = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]])
    emboss = cv2.filter2D(img, -1, kernel5)
    cv2.imshow("Emboss", emboss)
    cv2.waitKey(0)

def histogramEşitleme():
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]
    cv2.imshow('img', img2)
    plt.plot(cdf_normalized, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    cv2.waitKey(0)

def reSize():
    width = 300
    height = 600
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)

def rotate():
    # Window name in which image is displayed
    window_name = 'Image'

    # Using cv2.rotate() method
    # Using cv2.ROTATE_90_CLOCKWISE rotate
    # 90 derece döndürme
    image2 = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

    # Displaying the image
    cv2.imshow(window_name, image2)
    cv2.waitKey(0)

def crop():
    y = 0
    x = 0
    h = 300
    w = 510
    crop_image = img[x:w, y:h]
    cv2.imshow("Cropped", crop_image)
    cv2.waitKey(0)

def wrap():
    rows = img.shape[0]
    cols=img.shape[1]
    # Vertical wave
    img3 = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
            offset_y = 0
            if j+offset_x < rows:
                img3[i,j] = img[i,(j+offset_x)%cols]
            else:
                img3[i,j] = 0

    cv2.imshow('Wrap', img3)
    cv2.waitKey(0)

def affineWrap():
    rows, cols = img.shape[:2]

    src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0], [int(0.4*(cols-1)),rows-1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img4 = cv2.warpAffine(img, affine_matrix, (cols,rows))

    cv2.imshow('AffineWrap', img4)
    cv2.waitKey(0)

def intensityGamma():
    # Trying 4 gamma values.
    for gamma in [0.1, 0.5, 1.2, 2.2]:
        # Apply gamma correction.
        gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')

        # Save edited images.
        cv2.imshow("Gamma = "+str(gamma),gamma_corrected)
        cv2.waitKey(0)

def intensityLog():
    # Apply log transform.
    c = 255 / (np.log(1 + np.max(img)))
    log_transformed = c * np.log(1 + img)

    # Specify the data type.
    log_transformed = np.array(log_transformed, dtype=np.uint8)

    # Save the output.
    cv2.imshow('log_transformed', log_transformed)
    cv2.waitKey(0)

def erosion():
    kernel6 = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel6,iterations = 1)
    cv2.imshow("Erosion",erosion)
    cv2.waitKey(0)

def dilation():
    kernel7 = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel7, iterations=1)
    cv2.imshow("Dilation", dilation)
    cv2.waitKey(0)

def opening():
    kernel8 = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel8)
    cv2.imshow("Opening", opening)
    cv2.waitKey(0)

def closing():
    kernel9 = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel9)
    cv2.imshow("Closing", closing)
    cv2.waitKey(0)

def gradient():
    kernel10 = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel10)
    cv2.imshow("Gradient", gradient)
    cv2.waitKey(0)

def tophat():
    kernel11 = np.ones((5, 5), np.uint8)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel11)
    cv2.imshow("Tophat", tophat)
    cv2.waitKey(0)

def blackhat():
    kernel12 = np.ones((5, 5), np.uint8)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel12)
    cv2.imshow("BlackCat", blackhat)
    cv2.waitKey(0)

def rect():
    kernel13 = np.ones((5, 5), np.uint8)
    rect = cv2.morphologyEx(img, cv2.MORPH_RECT, kernel13)
    cv2.imshow("Rect", rect)
    cv2.waitKey(0)

def cross():
    kernel14 = np.ones((5, 5), np.uint8)
    cross = cv2.morphologyEx(img, cv2.MORPH_CROSS, kernel14)
    cv2.imshow("Cross", cross)
    cv2.waitKey(0)

def ellipse():
    kernel15 = np.ones((5, 5), np.uint8)
    ellipse = cv2.morphologyEx(img, cv2.MORPH_ELLIPSE, kernel15)
    cv2.imshow("Ellipse", ellipse)
    cv2.waitKey(0)

def activeCont():
    image5 = cv2.imread("image.png")
    # convert to RGB
    image5 = cv2.cvtColor(image5, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(image5, cv2.COLOR_RGB2GRAY)
    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    image = cv2.drawContours(image5, contours, -1, (0, 255, 0), 2)
    # show the image with the drawn contours
    plt.imshow(image5)
    plt.show()

def combineFilter():
    kernel16 = np.ones((5, 5), np.uint8)
    img6 =  cv2.dilate(img, kernel16, iterations=1)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(img6, -1, kernel)
    cv2.imshow("Sharpen", sharpen)
    cv2.waitKey(0)

def edgeDetect():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurr = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blurr, 10, 70)
        ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
        cv2.imshow('PRESS ENTER TO EXIT', mask)

        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()


#---------------------------TKINTERRRRRR-----------------------------------


root = Tk()
root.geometry('1100x500')

# Create a Button
btn = Button(root, text='Show Original Image', bd='5',command=showImg,fg="white", bg="black",pady=10, width=20, height=2)
btn.grid(row=0,column=0)

#-----------------LABEL----------------
var = StringVar()
label1  = Label( root, textvariable=var, relief=RAISED, width=15, height=2, bg="white")
var.set("Filters")
label1.grid(row=2,column=0,padx=10,pady=10)

var = StringVar()
label2  = Label( root, textvariable=var, relief=RAISED, width=15, height=2, bg="white")
var.set("Histogram")
label2.grid(row=2,column=2,padx=10,pady=10)

var = StringVar()
label3  = Label( root, textvariable=var, relief=RAISED, width=15, height=2, bg="white")
var.set("Uzaysal")
label3.grid(row=2,column=6,padx=10,pady=10)

var = StringVar()
label4  = Label( root, textvariable=var, relief=RAISED, width=15, height=2, bg="white")
var.set("Yoğunluk")
label4.grid(row=2,column=8,padx=10,pady=10)

var = StringVar()
label5 = Label( root, textvariable=var, relief=RAISED, width=15, height=2, bg="white")
var.set("Morfolojik")
label5.grid(row=2,column=10,padx=10,pady=10)

var = StringVar()
label6  = Label( root, textvariable=var, relief=RAISED, width=15, height=2, bg="white")
var.set("Active Contour")
label6.grid(row=2,column=12,padx=10,pady=10)

var = StringVar()
label7  = Label( root, textvariable=var, relief=RAISED, width=15, height=2, bg="white")
var.set("CombineFilter")
label7.grid(row=2,column=14,padx=10,pady=10)

var = StringVar()
label8 = Label( root, textvariable=var, relief=RAISED, width=15, height=2, bg="white")
var.set("Edge Detection")
label8.grid(row=2,column=16,padx=10,pady=10)

# ------BUTTONS1------

btn1 = Button(root, text='BoxFilter', bd='5',command=boxFilter,fg="black", bg="yellow", width=10, height=1)
btn1.grid(row=4,column=0)

btn2 = Button(root, text='Bilateral', bd='5',command=bilateralFilter,fg="black", bg="yellow", width=10, height=1)
btn2.grid(row=5,column=0)

btn3 = Button(root, text='HPF', bd='5',command=hpf,fg="black", bg="yellow", width=10, height=1)
btn3.grid(row=6,column=0)

btn4 = Button(root, text='LPF', bd='5',command=lpf,fg="black", bg="yellow", width=10, height=1)
btn4.grid(row=7,column=0)

btn5 = Button(root, text='Sharpen', bd='5',command=sharpen,fg="black", bg="yellow", width=10, height=1)
btn5.grid(row=8,column=0)

btn6 = Button(root, text='Sepia', bd='5',command=sepia,fg="black", bg="yellow", width=10, height=1)
btn6.grid(row=9,column=0)

btn7 = Button(root, text='Emboss', bd='5',command=boxFilter,fg="black", bg="yellow", width=10, height=1)
btn7.grid(row=10,column=0)

#-------------BUTTONS2-------------

btn8 = Button(root, text='Hist. Eşitleme', bd='5',command=histogramEşitleme,fg="black", bg="yellow", width=10, height=1)
btn8.grid(row=4,column=2)

#----------BUTTONS3-----------------

btn9 = Button(root, text='Resize', bd='5',command=reSize,fg="black", bg="yellow", width=10, height=1)
btn9.grid(row=4,column=6)

btn10 = Button(root, text='Rotate', bd='5',command=rotate,fg="black", bg="yellow", width=10, height=1)
btn10.grid(row=5,column=6)

btn11 = Button(root, text='Crop', bd='5',command=crop,fg="black", bg="yellow", width=10, height=1)
btn11.grid(row=6,column=6)

btn12 = Button(root, text='Wrap', bd='5',command=wrap,fg="black", bg="yellow", width=10, height=1)
btn12.grid(row=7,column=6)

btn13 = Button(root, text='AffineWrap', bd='5',command=affineWrap,fg="black", bg="yellow", width=10, height=1)
btn13.grid(row=8,column=6)

#----------BUTTONS4-------------------

btn14= Button(root, text='Intns.Gamma', bd='5',command=intensityGamma,fg="black", bg="yellow", width=10, height=1)
btn14.grid(row=4,column=8)

btn15 = Button(root, text='IntensityLog', bd='5',command=intensityLog,fg="black", bg="yellow", width=10, height=1)
btn15.grid(row=5,column=8)

#----------BUTTONS5-------------------

btn16= Button(root, text='Erosion', bd='5',command=erosion,fg="black", bg="yellow", width=10, height=1)
btn16.grid(row=4,column=10)

btn17 = Button(root, text='Dilation', bd='5',command=dilation,fg="black", bg="yellow", width=10, height=1)
btn17.grid(row=5,column=10)

btn18= Button(root, text='Opening', bd='5',command=opening,fg="black", bg="yellow", width=10, height=1)
btn18.grid(row=6,column=10)

btn19 = Button(root, text='Closing', bd='5',command=closing,fg="black", bg="yellow", width=10, height=1)
btn19.grid(row=7,column=10)

btn20= Button(root, text='Gradient', bd='5',command=gradient,fg="black", bg="yellow", width=10, height=1)
btn20.grid(row=8,column=10)

btn21 = Button(root, text='Tophat', bd='5',command=tophat,fg="black", bg="yellow", width=10, height=1)
btn21.grid(row=9,column=10)

btn22= Button(root, text='Blackhat', bd='5',command=blackhat,fg="black", bg="yellow", width=10, height=1)
btn22.grid(row=10,column=10)

btn23 = Button(root, text='Rect', bd='5',command=rect,fg="black", bg="yellow", width=10, height=1)
btn23.grid(row=11,column=10)

btn16= Button(root, text='Cross', bd='5',command=cross,fg="black", bg="yellow", width=10, height=1)
btn16.grid(row=12,column=10)

btn17 = Button(root, text='Ellipse', bd='5',command=ellipse,fg="black", bg="yellow", width=10, height=1)
btn17.grid(row=13,column=10)

#----------BUTTONS6-------------------

btn18= Button(root, text='ActiveCont', bd='5',command=activeCont,fg="black", bg="yellow", width=10, height=1)
btn18.grid(row=4,column=12)

#----------BUTTONS7-------------------

btn19= Button(root, text='CombineFilter', bd='5',command=combineFilter,fg="black", bg="yellow", width=10, height=1)
btn19.grid(row=4,column=14)

#----------BUTTONS8-------------------

btn20= Button(root, text='EdgeDetect', bd='5',command=edgeDetect,fg="black", bg="yellow", width=10, height=1)
btn20.grid(row=4,column=16)

root.mainloop()


