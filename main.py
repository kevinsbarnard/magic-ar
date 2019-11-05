# main.py
# Main runner for magic-ar app
import argparse
import cv2
import pytesseract
import numpy as np
import json
import difflib
from cv.detection import detect_rectangles, solve_poses

LOOP_DELAY = 30  # minimum delay in milliseconds
MAIN_WINDOW = 'Main Window'

row_crop = 0
col_crop = 0

all_cards = None


def row_crop_update(val):
    global row_crop
    row_crop = val


def col_crop_update(val):
    global col_crop
    col_crop = val


def setup():
    """
    Set up the application
    :return: None
    """
    cv2.namedWindow(MAIN_WINDOW)
    cv2.createTrackbar('Row Crop', MAIN_WINDOW, row_crop, 480 // 2, row_crop_update)
    cv2.createTrackbar('Column Crop', MAIN_WINDOW, col_crop, 640 // 2, col_crop_update)

    with open('AllCards.json', 'r') as f:
        global all_cards
        all_cards = json.load(f)


def teardown():
    """
    Tear down the application
    :return: None
    """
    cv2.destroyAllWindows()


def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def main():
    setup()

    cap = cv2.VideoCapture(0)

    rect_model = [
        [0, 0, 0],
        [2.5, 0, 0],
        [0, 3.5, 0],
        [2.5, 3.5, 0]
    ]

    camera_matrix = np.array(
        [
            [453, 0, 320],
            [0, 605, 240],
            [0, 0, 1]
        ], dtype=np.float32
    )

    axes = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32
    )

    card_titles = list(all_cards.keys())

    titlebox_ul = tuple([int(v) for v in np.array([0.125, 0.125]) * 200])
    titlebox_br = tuple([int(v) for v in np.array([2.375, 0.375]) * 200])

    while cv2.waitKey(LOOP_DELAY) != 27:
        ret, frame = cap.read()
        if not ret:
            break
        frame_h, frame_w, num_channels = frame.shape
        frame = frame[row_crop:frame_h-row_crop, col_crop:frame_w-col_crop]
        frame = cv2.flip(frame, -1)

        rect_contours, sorted_sets = detect_rectangles(frame)
        # cv2.drawContours(frame, rect_contours, -1, (0, 255, 0), -1)
        # cv2.drawContours(frame, rect_contours, -1, (0, 0, 255), 3)

        # for rect_set in sorted_sets:
        #     for corner in rect_set:
        #         cv2.circle(frame, tuple(corner), 3, (0, 0, 255), -1)

        sorted_sets = list(filter(lambda r_set: False if dist(r_set[0], r_set[2]) == 0 else 0.6 < dist(r_set[0], r_set[1]) / dist(r_set[0], r_set[2]) < 0.8, sorted_sets))
        poses = solve_poses(sorted_sets, rect_model, camera_matrix)

        perspective_transforms = []
        for rect_set in sorted_sets:
            rect_np = np.array(rect_set, dtype=np.float32)
            model_np = np.array(rect_model, dtype=np.float32)[:, :2]
            trans = cv2.getPerspectiveTransform(rect_np, model_np*200)
            perspective_transforms.append(trans)

        orthophotos = []
        for trans in perspective_transforms:
            warped = cv2.warpPerspective(frame, trans, (500, 700))
            # cv2.rectangle(warped, titlebox_ul, titlebox_br, (0, 255, 0), 2)
            orthophotos.append(warped)

        # hconc = cv2.hconcat(orthophotos)
        # if hconc is not None:
        #     cv2.imshow('orthophotos', hconc)

        for idx, ophoto in enumerate(orthophotos):
            roi = ophoto[titlebox_ul[1]:titlebox_br[1], titlebox_ul[0]:titlebox_br[0]]
            title = pytesseract.image_to_string(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            if title:
                close_matches = difflib.get_close_matches(title.strip(), card_titles, n=1)
                if close_matches:
                    matched_title = close_matches[0]
                    cv2.putText(frame, matched_title, sorted_sets[idx][0], cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                # print('Detected {}, close to {}'.format(title.strip(), close_matches))

        # for r_vec, t_vec in poses:
        #     axes_points, _ = cv2.projectPoints(axes, r_vec, t_vec, camera_matrix, np.zeros(4))
        #     cv2.line(frame, tuple(axes_points[0, 0, :]), tuple(axes_points[1, 0, :]), (0, 0, 255), 2)
        #     cv2.line(frame, tuple(axes_points[0, 0, :]), tuple(axes_points[2, 0, :]), (0, 255, 0), 2)
        #
        #     rm_pts_np = np.array(rect_model, dtype=np.float32)
        #     rm_points = _ = cv2.projectPoints(rm_pts_np, r_vec, t_vec, camera_matrix, np.zeros(4))
        #     cv2.line(frame, tuple(rm_points[0][0][0]), tuple(rm_points[0][3][0]), (255, 0, 0), 1)
        #     cv2.line(frame, tuple(rm_points[0][1][0]), tuple(rm_points[0][2][0]), (255, 0, 0), 1)

        cv2.imshow(MAIN_WINDOW, frame)

    teardown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
