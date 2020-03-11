import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]


# Initializing Parameters
v_var = 0.008  # translation velocity variance. Default: 0.01
om_var = 6 # rotational velocity variance. Default: 0.01
r_var = 0.00001  # range measurements variance. Default: 0.1
b_var = 0.001  # bearing measurement variance. Default: 0.1

Q_km = np.diag([v_var, om_var]) # input noise covariance
cov_y = np.diag([r_var, b_var])  # measurement noise covariance

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance


# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x


#
# ## Correction Step
def measurement_model(x_check, lk):
    x_check[2] = wraptopi(x_check[2])
    A = lk[0] - x_check[0, 0] - d[0] * np.cos(x_check[2, 0])
    B = lk[1] - x_check[1, 0] - d[0] * np.sin(x_check[2, 0])
    y_check = np.array([[np.sqrt(A ** 2 + B ** 2)], [np.arctan2 (B, A) - x_check[2, 0]]])
    y_check[1] = wraptopi(y_check[1])
    return y_check

def measurement_update(lk, rk, bk, P_check, x_check):
    # 1. Compute measurement Jacobian (size 2x3)
    x_check[2] = wraptopi(x_check[2])
    A = lk[0] - x_check[0, 0] - d[0] * np.cos(x_check[2, 0])
    B = lk[1] - x_check[1, 0] - d[0] * np.sin(x_check[2, 0])
    C = np.sqrt(A**2 + B**2)
    D = d[0] * np.sin(x_check[2, 0])
    E = d[0] * np.cos(x_check[2, 0])

    HK = np.array([[-A/C,   -B/C,    (A*D + B*(-E))/C],
                   [B/C**2, -A/C**2, -B/C**2*D + A/C**2*(-E) - 1]]) # HK size 2x3

    MK = np.eye(2)  # MK size 2x2

    # 2. Compute Kalman Gain
    RK = cov_y
    HPH = np.linalg.multi_dot([HK, P_check, HK.T])
    MRM = np.linalg.multi_dot([MK, RK, MK.T])
    KK = np.linalg.multi_dot([P_check, HK.T, np.linalg.inv(HPH + MRM)]) # kk size 3x2

    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    y_check = measurement_model(x_check, lk)
    y_check[1] = wraptopi(y_check[1])
    bk = wraptopi(bk)
    y_act = np.array([[rk], [bk]])
    dy = y_act - y_check
    adder = np.dot(KK, dy)
    x_check = x_check + adder
    x_check[2] = wraptopi(x_check[2])

    # 4. Correct covariance
    P_check = np.dot((np.eye(3) - KK.dot(HK)), P_check) # P_check size 3x3

    return x_check, P_check


# Prediction Step
def motion_model(x_check, k, delta_t):
    A = np.array([[np.cos(x_check[2, 0]), 0], [np.sin(x_check[2, 0]), 0], [0, 1]])
    #A = np.array([[np.cos(x_check[2]), 0], [np.cos(x_check[2]), 0], [0, 1]])
    B = np.array([[v[k]], [om[k]]])
    adder = delta_t * np.dot(A, B)
    x_check = x_check + adder
    x_check[2] = wraptopi(x_check[2])
    return x_check

# Main Filter Loop
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    x_check = np.zeros([3, 1])
    x_check = motion_model(np.array([x_est[k-1]]).T, k, delta_t)

    # 2. Motion model jacobian with respect to last state
    #F_km = np.zeros([3, 3])
    F_km = np.eye(3)
    F_km[:, 2] = F_km[:, 2] + np.array([-delta_t*v[k]*np.sin(x_est[k-1, 2]),
                                       delta_t*v[k]*np.cos(x_est[k-1, 2]),
                                       0])

    # 3. Motion model jacobian with respect to noise
    L_km = np.zeros([3, 2])
    L_km = delta_t * np.array([[np.cos(x_est[k-1, 2]), 0], [np.sin(x_est[k-1, 1]), 0], [0, 1]])

    # 4. Propagate uncertainty
    noise = np.linalg.multi_dot([L_km, Q_km, L_km.T])
    motion = np.linalg.multi_dot([F_km, P_est[k-1,:,:], F_km.T])
    P_check =  motion + noise

    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check


# Let's plot the resulting state estimates:
e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()


with open('submission.pkl', 'wb') as f:
    pickle.dump(x_est, f, pickle.HIGHEST_PROTOCOL)
