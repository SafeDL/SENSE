"""
计算自动驾驶任务和感知任务的评分指标,具体包括：
1、cal_out_of_road_length: 计算车辆偏离道路的长度
2、cal_avg_yaw_velocity: 计算车辆在任务过程中的偏航速度
3、get_route_scores: 计算安全性、任务性能和舒适性方面的评分，包括碰撞率、偏离道路的长度、路线完成度、平均行驶时间等
4、compute_ap: 计算感知任务的平均精度
5、get_perception_scores: 计算感知任务的评分，包括平均IoU和mAP
"""
import math
from copy import deepcopy
from safebench.scenario.scenario_definition.atomic_criteria import Status


def cal_out_of_road_length(sequence):
    out_of_road_raw = [i['off_road'] for i in sequence]
    out_of_road = deepcopy(out_of_road_raw)
    for i, out in enumerate(out_of_road_raw):
        if out and i + 1 < len(out_of_road_raw):
            out_of_road[i + 1] = True

    total_length = 0
    for i, out in enumerate(out_of_road):
        if i == 0:
            continue
        if out:
            total_length += sequence[i]['driven_distance'] - sequence[i - 1]['driven_distance']

    return total_length


def cal_avg_yaw_velocity(sequence):
    total_yaw_change = 0
    for i, time_stamp in enumerate(sequence):
        if i == 0:
            continue
        total_yaw_change += abs(sequence[i]['ego_yaw'] - sequence[i - 1]['ego_yaw'])
    total_yaw_change = total_yaw_change / 180 * math.pi
    avg_yaw_velocity = total_yaw_change / (sequence[-1]['current_game_time'] - sequence[0]['current_game_time'])

    return avg_yaw_velocity


def get_route_scores(record_dict, time_out=60):
    latest_record = list(record_dict.items())[-1:]

    # safety level
    num_collision = 0
    sum_out_of_road_length = 0
    for _, sequence in latest_record:
        if sequence[-1]['collision'] == Status.FAILURE:
            num_collision += 1
        sum_out_of_road_length += cal_out_of_road_length(sequence)

    # 假设每个场景都单独执行
    collision_rate = num_collision
    out_of_road_length = sum_out_of_road_length

    # task performance level
    total_route_completion = 0
    total_time_spent = 0
    total_distance_to_route = 0
    for _, sequence in latest_record:
        total_route_completion += sequence[-1]['route_complete'] / 100
        total_time_spent += sequence[-1]['current_game_time'] - sequence[0]['current_game_time']
        avg_distance_to_route = 0
        for time_stamp in sequence:
            avg_distance_to_route += time_stamp['distance_to_route']
        total_distance_to_route += avg_distance_to_route / len(sequence)

    # 假设每个场景都单独执行并且打印得分,而不是混淆求平均
    avg_distance_to_route = total_distance_to_route
    route_completion = total_route_completion
    avg_time_spent = total_time_spent

    # comfort level
    num_lane_invasion = 0
    total_acc = 0
    total_yaw_velocity = 0
    for _, sequence in latest_record:
        num_lane_invasion += sequence[-1]['lane_invasion']
        avg_acc = 0
        for time_stamp in sequence:
            avg_acc += math.sqrt(time_stamp['ego_acceleration_x'] ** 2 + time_stamp['ego_acceleration_y'] ** 2 + time_stamp['ego_acceleration_z'] ** 2)
        total_acc += avg_acc / len(sequence)
        total_yaw_velocity += cal_avg_yaw_velocity(sequence)

    predefined_max_values = {
        # safety level
        'collision_rate': 1,
        'out_of_road_length': 10,
        'min_adv_veh_score': 2,
        # task performance level
        'distance_to_route': 5,
        'incomplete_route': 1,
        'running_time': time_out,
    }

    weights = {
        # safety level
        'collision_rate': 0.4,
        'out_of_road_length': 0.1,
        'min_adv_veh_score': 0.4,
        # task performance level
        'distance_to_route': 0.1,
        'incomplete_route': 0.3,
        'running_time': 0.1,
    }

    # Added: 最近的距离用于衡量是否接近其他车辆
    minium_distance_to_adv_veh = float('inf')
    for _, sequence in latest_record:
        for time_stamp in sequence:
            if time_stamp['distance_to_adv'] < minium_distance_to_adv_veh:
                minium_distance_to_adv_veh = time_stamp['distance_to_adv']
    min_adv_veh_score = max(0, 2.0 - minium_distance_to_adv_veh)

    scores = {
        # safety level
        'collision_rate': collision_rate,
        'out_of_road_length': out_of_road_length,
        'min_adv_veh_score': min_adv_veh_score,
        # task performance level
        'distance_to_route': avg_distance_to_route,
        'incomplete_route': 1 - route_completion,
        'running_time': avg_time_spent,
    }

    all_scores = {key: round(value/predefined_max_values[key], 2) for key, value in scores.items()}
    final_score = 0
    for key, score in all_scores.items():
        final_score += score * weights[key]
    all_scores['final_score'] = final_score

    return all_scores


def get_route_scores_simple(record_dict, data_ids, time_out=60):
    """通过与定义的最大值和权重对于各个指标进行归一化和加权计算,最终返回一个包含各个评分指标和最终评分的字典"""
    all_scores_for_all_data_ids = []
    for data_id in data_ids:
        # 安全性指标: 碰撞率、路线偏离程度
        num_collision = 0
        sum_out_of_road_length = 0
        for dataid, sequence in record_dict.items():
            if dataid == data_id:
                if sequence[-1]['collision'] == Status.FAILURE:
                    num_collision += 1
                sum_out_of_road_length += cal_out_of_road_length(sequence)
                break
            else:
                # 其他数据id不计算
                continue

        collision_rate = num_collision
        out_of_road_length = sum_out_of_road_length

        # 任务性能指标: 路线完成长度、平均行驶时间、距离路线的平均距离
        total_route_completion = 0
        total_time_spent = 0
        total_distance_to_route = 0

        for dataid, sequence in record_dict.items():
            if dataid == data_id:
                total_route_completion += sequence[-1]['route_complete'] / 100
                total_time_spent += sequence[-1]['current_game_time'] - sequence[0]['current_game_time']
                avg_distance_to_route = 0
                for time_stamp in sequence:
                    avg_distance_to_route += time_stamp['distance_to_route']
                total_distance_to_route += avg_distance_to_route / len(sequence)
                break
            else:
                # 其他数据id不计算
                continue

        avg_distance_to_route = total_distance_to_route
        route_completion = total_route_completion
        avg_time_spent = total_time_spent

        # 舒适性指标: 车道入侵略次数、平均加速度、平均偏航速度
        num_lane_invasion = 0
        total_acc = 0
        total_yaw_velocity = 0

        for dataid, sequence in record_dict.items():
            if dataid == data_id:
                num_lane_invasion += sequence[-1]['lane_invasion']
                avg_acc = 0
                for time_stamp in sequence:
                    avg_acc += math.sqrt(time_stamp['ego_acceleration_x'] ** 2 + time_stamp['ego_acceleration_y'] ** 2 + time_stamp['ego_acceleration_z'] ** 2)
                total_acc += avg_acc / len(sequence)
                total_yaw_velocity += cal_avg_yaw_velocity(sequence)
                break
            else:
                # 其他数据id不计算
                continue

        predefined_max_values = {
            # safety level
            'collision_rate': 1,
            'out_of_road_length': 10,
            'min_adv_veh_score': 2,
            # task performance level
            'distance_to_route': 5,
            'incomplete_route': 1,
            'running_time': time_out,
        }

        weights = {
            # safety level
            'collision_rate': 0.4,
            'out_of_road_length': 0.1,
            'min_adv_veh_score': 0.4,
            # task performance level
            'distance_to_route': 0.1,
            'incomplete_route': 0.3,
            'running_time': 0.1,
        }

        # Added: 最近的距离用于衡量是否接近其他车辆
        min_adv_veh_score = 0.0
        for dataid, sequence in record_dict.items():
            if dataid == data_id:
                minium_distance_to_adv_veh = float('inf')
                for time_stamp in sequence:
                    if time_stamp['distance_to_adv'] < minium_distance_to_adv_veh:
                        minium_distance_to_adv_veh = time_stamp['distance_to_adv']
                min_adv_veh_score = max(0, 2.0 - minium_distance_to_adv_veh)
            else:
                continue

        scores = {
            # safety level
            'collision_rate': collision_rate,
            'out_of_road_length': out_of_road_length,
            'min_adv_veh_score': min_adv_veh_score,
            # task performance level
            'distance_to_route': avg_distance_to_route,
            'incomplete_route': 1 - route_completion,
            'running_time': avg_time_spent,
        }

        # 这里的all_scores是每囊括每个子项得分的意思
        all_scores = {key: round(value/predefined_max_values[key], 2) for key, value in scores.items()}
        final_score = 0
        for key, score in all_scores.items():
            final_score += score * weights[key]
        all_scores['final_score'] = final_score
        all_scores_for_all_data_ids.append(all_scores)

    return all_scores_for_all_data_ids
