# main.py
# Main runner for magic-ar app
import argparse
import cv2
import pytesseract
import numpy as np
import json
import difflib
from cv.detection import detect_rectangles
import textwrap
import unidecode

LOOP_DELAY = 30  # minimum delay in milliseconds
MAIN_WINDOW = 'Main Window'

row_crop = 0
col_crop = 0
focus = 0

all_cards = None


def row_crop_update(val):
    global row_crop
    row_crop = val


def col_crop_update(val):
    global col_crop
    col_crop = val


def focus_update(val):
    global focus
    focus = val


def setup():
    """
    Set up the application
    :return: None
    """
    cv2.namedWindow(MAIN_WINDOW)
    cv2.createTrackbar('Row Crop', MAIN_WINDOW, row_crop, 480 // 2, row_crop_update)
    cv2.createTrackbar('Column Crop', MAIN_WINDOW, col_crop, 640 // 2, col_crop_update)
    cv2.createTrackbar('Focus', MAIN_WINDOW, focus, 255, focus_update)

    with open('AllCards.json', 'r') as f:
        global all_cards
        all_cards = json.load(f)


def teardown():
    """
    Tear down the application
    :return: None
    """
    cv2.destroyAllWindows()


def dist(p1: iter, p2: iter):
    """
    Compute Euclidean distance between two points.
    :param p1: First point iterable
    :param p2: Second point iterable
    :return: Float distance
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_closest_pt(t_map, pt):
    min_dist_pt = None
    min_dist = 999999
    for pt_close, pt_title in t_map.items():
        dist_close = dist(pt, pt_close)
        if dist_close < min_dist:
            min_dist = dist_close
            min_dist_pt = pt_close
    return min_dist_pt


def get_translation_info(name, language):
    for trans_item in all_cards[name]['foreignData']:
        if trans_item['language'] == language:
            if 'name' in trans_item.keys() and 'text' in trans_item.keys():
                return trans_item['name'], trans_item['text']


def main():
    setup()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, False)

    rect_model = [
        [0, 0, 0],
        [2.5, 0, 0],
        [0, 3.5, 0],
        [2.5, 3.5, 0]
    ]

    card_titles = list(all_cards.keys())

    titlebox_ul = tuple([int(v) for v in np.array([0.125, 0.125]) * 200])
    titlebox_br = tuple([int(v) for v in np.array([2.375, 0.375]) * 200])

    title_map = dict()

    fourcc_str = 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vid_out = cv2.VideoWriter('vid_out.avi', fourcc, 10, (640, 480))

    frame_num = 0

    while cv2.waitKey(LOOP_DELAY) != 27:
        # Read in from the video source
        cap.set(cv2.CAP_PROP_FOCUS, focus)
        ret, frame = cap.read()
        if not ret:
            break
        frame_h, frame_w, num_channels = frame.shape
        frame = frame[row_crop:frame_h - row_crop, col_crop:frame_w - col_crop]
        frame = cv2.flip(frame, -1)

        # Detect rectangles
        rect_contours, sorted_sets = detect_rectangles(frame)

        # Filter out any rectangles whose ratio of width/height is out of bounds
        sorted_sets = list(filter(
            lambda r_set: False if dist(r_set[0], r_set[2]) == 0 else 0.5 < dist(r_set[0], r_set[1]) / dist(r_set[0], r_set[2]) < 0.9,
            sorted_sets
        ))

        # Solve perspective transforms (image -> model)
        perspective_transforms = []
        for rect_set in sorted_sets:
            rect_np = np.array(rect_set, dtype=np.float32)
            model_np = np.array(rect_model, dtype=np.float32)[:, :2]
            trans = cv2.getPerspectiveTransform(rect_np, model_np * 200)
            perspective_transforms.append(trans)

        # Generate orthophotos
        orthophotos = []
        for trans in perspective_transforms:
            warped = cv2.warpPerspective(frame, trans, (500, 700))
            cv2.rectangle(warped, titlebox_ul, titlebox_br, (0, 255, 0), 2)
            orthophotos.append(warped)

        # Identify, translate, and reproject onto each card
        for idx, (ophoto, trans) in enumerate(zip(orthophotos, perspective_transforms)):
            # Get title box from orthophoto
            roi = ophoto[titlebox_ul[1]:titlebox_br[1], titlebox_ul[0]:titlebox_br[0]]

            # Run ROI through tesseract
            title = pytesseract.image_to_string(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), config=('-l eng --oem 1 --psm 3'))
            matched_title = ''

            # Make a blank canvas
            canvas = np.zeros((int(3.5 * 200 * 3), int(2.5 * 200 * 3), 3), dtype=np.uint8)
            if title:
                title = (''.join(filter(lambda c: c.isalpha() or c == ' ', title))).strip()
                close_matches = difflib.get_close_matches(title, card_titles, n=1)
                if close_matches:
                    matched_title = close_matches[0]
                    closest_pt = get_closest_pt(title_map, sorted_sets[idx][0])
                    if closest_pt and dist(closest_pt, sorted_sets[idx][0]) < 100:
                        title_map.pop(closest_pt)
                    title_map[sorted_sets[idx][0]] = matched_title
                else:
                    closest_pt = get_closest_pt(title_map, sorted_sets[idx][0])
                    if closest_pt and dist(closest_pt, sorted_sets[idx][0]) < 1000:
                        matched_title = title_map[closest_pt]
            else:
                closest_pt = get_closest_pt(title_map, sorted_sets[idx][0])
                if closest_pt and dist(closest_pt, sorted_sets[idx][0]) < 1000:
                    matched_title = title_map[closest_pt]

            # Lookup card info and paint the canvas
            if matched_title:
                trans_data = get_translation_info(matched_title, 'Spanish')
                if not trans_data:
                    continue
                trans_name = trans_data[0]
                trans_name = unidecode.unidecode(trans_name)
                trans_text = trans_data[1]
                trans_text = unidecode.unidecode(trans_text)
                wrap_text = textwrap.wrap(trans_text, 20)

                # Put the card title
                cv2.putText(
                    canvas, trans_name,
                    (25, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=3
                )

                # Put the card description
                for line_idx, line in enumerate(wrap_text):
                    cv2.putText(
                        canvas, line,
                        (25, 50 * (line_idx + 3)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), thickness=3
                    )

                warped_canvas = cv2.warpPerspective(canvas, trans, (frame_w, frame_h), flags=cv2.WARP_INVERSE_MAP)

                _, canvas_mask = cv2.threshold(
                    cv2.cvtColor(warped_canvas, cv2.COLOR_BGR2GRAY),
                    0, 255,
                    cv2.THRESH_BINARY_INV
                )
                canvas_mask = cv2.cvtColor(canvas_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                frame = cv2.bitwise_and(frame, canvas_mask)
                frame = cv2.bitwise_or(frame, warped_canvas)

        cv2.imshow(MAIN_WINDOW, frame)
        vid_out.write(frame)
        frame_num += 1
        cv2.imwrite('frames/frame_{}.png'.format(frame_num), frame)

    vid_out.release()
    teardown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
