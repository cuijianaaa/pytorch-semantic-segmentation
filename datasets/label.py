        self.id_to_trainid = {
            -1: ignore_label,#没用
            1: ignore_label, #//自我交通工
            2: ignore_label, #//矫正边缘
            3: ignore_label, #//roi之外
            4: ignore_label, #//静态
            5: ignore_label, #//动态
            6: ignore_label, #//ground
            7: 0,            #//road
            8: 1,            #//人行道
            9: ignore_label, #//停车位
            10: ignore_label,#//铁路
            11: 2,           #//建筑
            12: 3,           #//墙
            13: 4,           #//栅栏
            14: ignore_label,#//护栏
            15: ignore_label,#//桥
            16: ignore_label,#//隧道
            17: 5,           #//杆子
            18: ignore_label,#//一堆杆子
            19: 6,           #//交通灯
            20: 7,           #//交通标志
            21: 8,           #//植物
            22: 9,           #//地形，地面
            23: 10,          #//天空
            24: 11,          #//人
            25: 12,          #//骑行者
            26: 13,          #//car
            27: 14,          #//truck
            28: 15,          #//bus
            29: ignore_label,#//篷车
            30: ignore_label,#//拖车
            31: 16,          #//train
            32: 17,          #//摩托车
            33: 18}          #//自行车