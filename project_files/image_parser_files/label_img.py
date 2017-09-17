def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat':
        return 1
    #                             [no cat, very doggo]
    elif word_label == 'dog':
        return 0
