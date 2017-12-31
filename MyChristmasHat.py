import cv2
import dlib

hat_img_dir="./data//hat.png"
hat_alpha_dir="./result/alpha.jpg"
hat_rgb_dir="./result/rgb.jpg"

def wear_a_hat(img, hat_img):
    
    g,b,r,a = cv2.split(hat_img)
    hat_gbr = cv2.merge((g,b,r))
    # cv2.imwrite(hat_rgb_dir, hat_gbr)
    # cv2.imwrite(hat_alpha_dir, a)

    # detect key points of a face 
    predict_path="./data/shape_predictor_5_face_landmarks.dat"
    predicter=dlib.shape_predictor(predict_path)

    detector=dlib.get_frontal_face_detector()

    faces = detector(img,1)

    if len(faces)>0:
        for face in faces:
            # since the coordinates of an image is from (0,0) top left to botom right
            x,y,w,h = face.left(), face.top(), face.right()-face.left(), face.bottom()-face.top()
            # cv2.rectangle(img, pt1=(x,y), pt2=(x+w, y+h),color=(255,0,0), thickness=2, lineType=8, shift=0)

            keypoints = predicter(img, face)
            # # showing the key point out
            # for point in keypoints.parts():
            #     cv2.circle(img, center=(point.x, point.y), radius=3, color=(0,255,0))

            # cv2.imshow("image", img)
            # cv2.waitKey()
            left_eye_point = keypoints.part(0)
            right_eye_point = keypoints.part(2)
            # notice that //: divide with integral result (discard remainder)
            # and also it is adding then divide instead of substract then divide
            center_eye_point = ((right_eye_point.x+left_eye_point.x)//2, (right_eye_point.y+left_eye_point.y)//2)
            # adjust the hat size: hat_gbr.shape[0] stands for the width of image, hat_gbr.shape[1] stands for the height of the image
            factor=1.5
            resized_hat_height=int(hat_gbr.shape[0]*w/hat_gbr.shape[1]*factor)
            resized_hat_width=int(hat_gbr.shape[1]*w/hat_gbr.shape[1]*factor)
            resized_rgb_hat = cv2.resize(hat_gbr,dsize=(resized_hat_width, resized_hat_height))

            # use alpha channel as the mask
            mask = cv2.resize(a, dsize=(resized_hat_width, resized_hat_height))
            # black white exchange
            mask_inv = cv2.bitwise_not(src=mask)

            relative_height_offset=0
            relative_width_offset=0

            roi = img[y+relative_height_offset-resized_hat_height:y+relative_height_offset, 
                (center_eye_point[0]-resized_hat_width//3):(center_eye_point[0]+resized_hat_width//3*2)]
            roi = roi.astype(float)

            # extract the ROI in the origin image for placing the hat. But why in this way?
            mask_inv=cv2.merge(mv=(mask_inv, mask_inv, mask_inv))
            # I need to let the other part to be 0 which is white so that I can add to the origin image
            alpha = mask_inv.astype(float)/255
            alpha = cv2.resize(alpha, dsize=(roi.shape[1], roi.shape[0]))
            print("alpha shape:", alpha.shape)
            print("roi shape:", roi.shape)
            bg = cv2.multiply(alpha, roi)
            bg = bg.astype('uint8')

            # cv2.imwrite("bg.jpg",bg)
            cv2.imshow("bg.jpg",bg)
            cv2.waitKey()
            
            # extract the hat
            hat = cv2.bitwise_and(resized_rgb_hat, resized_rgb_hat, mask=mask)
            cv2.imshow("hat.jpg",hat)
            cv2.waitKey()

            # make sure the hat has the same size as the roi
            hat = cv2.resize(hat, dsize=(roi.shape[1], roi.shape[0]))
            print(bg.shape)
            print(hat.shape)
            added_hat = cv2.add(bg, hat)
            img[y+relative_height_offset-resized_hat_height:y+relative_height_offset, 
                (center_eye_point[0]-resized_hat_width//3):(center_eye_point[0]+resized_hat_width//3*2)] = added_hat
            cv2.imshow("image", img)
            cv2.waitKey()
            return img


if __name__ == "__main__":
    test_img=cv2.imread("./data/me.jpg")
    hat_img = cv2.imread(hat_img_dir, cv2.IMREAD_UNCHANGED)
    result = wear_a_hat(test_img, hat_img)
    cv2.imwrite("./result/i-got-an-hat.jpg", img=result)
    cv2.destroyAllWindows()









