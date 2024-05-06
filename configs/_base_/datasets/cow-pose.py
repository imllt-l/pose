dataset_info = dict(
    dataset_name='cow-pose',
    paper_info=dict(
        author='imllt',
        title='cow-pose',
        year='2024',
    ),
    keypoint_info={
        0:
        dict(name='Head_top', id=0, color=[240, 0, 0], type='upper', swap=''),
        1:
        dict(
            name='Neck',
            id=1,
            color=[204, 168, 168],
            type='upper',
            swap=''),        
        2:
        dict(
            name='Spine',
            id=2,
            color=[166, 88, 187],
            type='upper',
            swap=''),
        3:
        dict(
            name='Right_FTR',
            id=3,
            color=[31, 96, 51],
            type='lower',
            swap='Left_FTR'),
        4:
        dict(
            name='Right_FK',
            id=4,
            color=[48, 153, 41],
            type='lower',
            swap='Left_FK'),
        5:
        dict(
            name='Right_FH',
            id=5,
            color=[79, 56, 112],
            type='lower',
            swap='Left_FH'),
        6:
        dict(
            name='Left_FTR',
            id=6,
            color=[159, 214, 207],
            type='lower',
            swap='Right_FTR'),
        7:
        dict(
            name='Left_FK',
            id=7,
            color=[40, 195, 115],
            type='lower',
            swap='Right_FK'), 
        8:
        dict(
            name='Left_FH',
            id=8,
            color=[93, 223, 221],
            type='lower',
            swap='Right_FH'),      
        9:
        dict(
            name='Coccyx',
            id=9,
            color=[37, 165, 208],
            type='upper',
            swap=''),
        10:
        dict(
            name='Right_HT',
            id=10,
            color=[255, 117, 117],
            type='lower',
            swap='Left_HTR'),
        11:
        dict(
            name='Right_HK',
            id=11,
            color=[255, 255, 255],
            type='lower',
            swap='Left_HK'),
        12:
        dict(
            name='Right_HH',
            id=12,
            color=[233, 226, 226],
            type='lower',
            swap='Left_HH'),
        13:
        dict(
            name='Left_HTR',
            id=13,
            color=[0, 0, 0],
            type='lower',
            swap='Right_HT'),
        14:
        dict(
            name='Left_HK',
            id=14,
            color=[0, 0, 0],
            type='lower',
            swap='Right_HK'),
        15:
        dict(
            name='Left_HH',
            id=15,
            color=[0, 0, 0],
            type='lower',
            swap='Right_HH'),
        16:
        dict(
            name='Test',
            id=16,
            color=[0, 0, 0],
            type='',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('Neck'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('Head_top','Spine'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('Neck','Coccyx','Left_FTR','Right_FTR'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('Spine','Right_FK'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('Right_FTR','Right_FH'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('Right_FK'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('Spine','Left_FK'), id=6, color=[0, 255, 0]),
        7:
        dict(link=('Left_FH','Left_FTR'), id=7, color=[0, 255, 0]),
        8:
        dict(link=('Left_FK'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('Spine','Right_HT','Left_HTR'), id=9, color=[0, 255, 0]),
        10:
        dict(link=('Coccyx','Right_HK'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('Right_HT','Right_HH'), id=11, color=[0, 255, 0]),
        12:
        dict(link=('Right_HK'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('Coccyx','Left_HK'), id=13, color=[0, 255, 0]),
        14:
        dict(link=('Left_HTR','Left_HH'), id=14, color=[0, 255, 0]),
        15:
        dict(link=('Left_HK'), id=15, color=[0, 255, 0]),
    },
    joint_weights=[
        1., 1., 1., 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1., 1.5, 1.2, 1.2, 1.2, 1.2, 1.2,
        0
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0
    ]
)