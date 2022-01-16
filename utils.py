import numpy as np


def _is_box_valid(box, roi_dims, area_prcntg=0.1):  # default area of the face should be at least 10% the area of ROI
    x, y, width, height = box
    x_ctr, y_ctr = round(x+(width/2)), round(y+(height/2))
    roi_x, roi_y, roi_width, roi_height = roi_dims
    if (x_ctr<roi_x) or (y_ctr<roi_y) or (x_ctr>(roi_x+roi_width)) or (y_ctr>(roi_y+roi_height)):
        return False, 'bounding box center outside the ROI'
    if (width*height) < area_prcntg*(roi_width*roi_height):
        return False, 'bounding box area smaller than '+str(area_prcntg)+'* ROI area'
    else:
        return True, None


def _calc_eucl_dist(point1, point2):
    return np.sqrt(np.sum(np.square(np.array(point1) - np.array(point2))))


def get_roi(raw_img, roi_cut):
    roi = [round(roi_cut*np.shape(raw_img)[0]/2), round(roi_cut*np.shape(raw_img)[1]/2),
            round(np.shape(raw_img)[1]-roi_cut*np.shape(raw_img)[1]),
            round(np.shape(raw_img)[0]-roi_cut*np.shape(raw_img)[0])]
    return roi


def get_valid_bboxes(bboxes, roi_dims):
    valid_bboxes = []
    for box in bboxes:
        valid, status = _is_box_valid(box, roi_dims)
        if valid:
            valid_bboxes.append(box)
    return np.array(valid_bboxes)


def get_focused_box(bboxes, img_dims):
    # focused based on the distance of center of bbox to the center of image
    x_im_ctr, y_img_ctr = round(img_dims[1]/2), round(img_dims[0]/2) # np.shape(image) is (height, width), not (width, height)
    min_eucl_dist = _calc_eucl_dist([x_im_ctr, y_img_ctr], [img_dims[1], img_dims[0]])  # assigned the hypothetica; farthest possible bbox
    focus_box = []
    for box in bboxes:
        x, y, width, height = box
        x_ctr, y_ctr = round(x+(width/2)), round(y+(height/2))
        eucl_dist = _calc_eucl_dist([x_im_ctr, y_img_ctr], [x_ctr, y_ctr])
        if  eucl_dist < min_eucl_dist:
            focus_box = box
            min_eucl_dist = eucl_dist
    focus_box = np.array(focus_box)
    focus_box = np.reshape(focus_box, (1,4))
    return focus_box


def disp_msg():
    print("""
>>>Detecting Faces in the ROI,
>>>Focusing on the face close to the center of the video frame,
>>>Detecting 68 Facial landmarks,
>>>Detecting and Tracking facial pose
  .
  .
  .
>>>Press Cntl+C to stop the program
          """)
