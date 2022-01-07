top_left_a = (0, 0)
bottom_right_a = (1, 1)

top_left_b = (0, 0)
bottom_right_b = (0, 0)

if top_left_a[0] > bottom_right_b[0] or \
        bottom_right_a[0] < top_left_b[0] or \
        top_left_a[1] > bottom_right_b[1] or \
        bottom_right_a[1] < top_left_b[1]:
    print('non')
