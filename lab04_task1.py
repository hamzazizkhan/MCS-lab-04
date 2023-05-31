import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import matplotlib
'''
increase arrow size for better visibility!
'''
def z1(pt, center_of_mass):
    if center_of_mass is None:
        return 0
    repulsion_vector = -(center_of_mass - pt)
    theta1 = np.arctan2(repulsion_vector[1], repulsion_vector[1]) - np.arctan2(pt[1], pt[0])
    return theta1

def z2(neighbourhood):
    if len(neighbourhood) == 0:
        return 0
    angles = [np.arctan2(pt_neigh[1], pt_neigh[0]) for pt_neigh in neighbourhood]
    theta2 = sum(angles)/len(angles)
    return theta2

def z3(pt, center_of_mass):
    if center_of_mass is None:
        return 0
    att_vector = (center_of_mass - pt)
    theta1 = np.arctan2(att_vector[1], att_vector[0]) - np.arctan2(pt[1], pt[0])
    return theta1



n = 50
seed_value = 42
random.seed(seed_value)
x = [random.uniform(-7, 7) for _ in range(n)]
y = [random.uniform(-7, 7) for _ in range(n)]
xy = list(zip(x, y))

data = []
data.append(np.array(xy))
velocity = []
velocity_vectors =[]
for pt in xy:
    pt = np.array(pt)
    velocity_vector = 1 / np.linalg.norm(pt) * pt
    velocity_vectors.append(tuple(velocity_vector))
velocity.append(velocity_vectors)
#print(data)

# manipulate n, initial dist of pts,  radius, frames, rho, neigh_rad
frames = 200
radius = [5, 10, 15]
rho = [0.2,0.3,0.4,0.1]
ab = [0.2,-1.2]

cm = []
sumz2ne = 0
sumz3ne = 0
for frame in range(frames):
    nc_z1 = np.array([(k, v) for k, v in xy if k ** 2 + v ** 2 <= radius[0]**2])
    if len(nc_z1) != 0:
        centre_mass_z1 = np.array(sum(nc_z1) / len(nc_z1))
    if len(nc_z1) == 0:

        centre_mass_z1 = None


    nc_z2 = np.array([(k, v) for k, v in xy if k ** 2 + v ** 2 <= radius[1]**2])

    nc_z3 = np.array([(k, v) for k, v in xy if k ** 2 + v ** 2 <= radius[2]**2])
    if len(nc_z3) != 0:
        centre_mass_z3 = np.array(sum(nc_z3) / len(nc_z3))
    if len(nc_z3) == 0:
        centre_mass_z3 = None
    cm.append((centre_mass_z3))

    new_xy = []
    new_velocity = []
    for pt in xy:
        # zone1
        pt = np.array(pt)
        velocity_vector = 1 / np.linalg.norm(pt) * pt


        theta1 = z1(pt, centre_mass_z1)
        rotation_matrix = np.array([[np.cos(theta1), -np.sin(theta1)],
                                    [np.sin(theta1), np.cos(theta1)]])
        v_rotated_theta1 = np.dot(rotation_matrix, velocity_vector)


        '''new_theta = theta1*rho[0]
        rotation_matrix = np.array([[np.cos(new_theta), -np.sin(new_theta)],
                                    [np.sin(new_theta), np.cos(new_theta)]])
        v_rotated = np.dot(rotation_matrix, velocity_vector)
        newpt = pt + v_rotated
        #print(v_rotated, velocity_vector, 'here')
        new_xy.append(tuple(newpt))
        new_velocity.append(tuple(v_rotated))'''

        # zone 2
        neigh_rad = radius[1]

        neighbourhood = [(point) for point in nc_z2 if tuple(point) != tuple(pt) if
                             (point[0] - pt[0]) ** 2 + (point[1] - pt[1]) ** 2 <= neigh_rad]
        theta2 = z2(neighbourhood)
        rotation_matrix = np.array([[np.cos(theta2), -np.sin(theta2)],
                                    [np.sin(theta2), np.cos(theta2)]])
        v_rotated_theta2 = np.dot(rotation_matrix, velocity_vector)

        if theta2==0:
            print('True, particle has no neigh in z2')
            sumz2ne += 1


        '''rotation_matrix = np.array([[np.cos(theta2), -np.sin(theta2)],
                                    [np.sin(theta2), np.cos(theta2)]])
        v_rotated = np.dot(rotation_matrix, velocity_vector)
        newpt = pt + v_rotated

        new_xy.append(tuple(newpt))
        new_velocity.append(tuple(v_rotated))'''
        # zone 3
        theta3 = z3(pt, centre_mass_z3)

        rotation_matrix = np.array([[np.cos(theta3), -np.sin(theta3)],
                                    [np.sin(theta3), np.cos(theta3)]])
        v_rotated_theta3 = np.dot(rotation_matrix, velocity_vector)

        if theta3==0:
            print('True, particle has EXITED Z3')
            sumz3ne += 1

        '''rotation_matrix = np.array([[np.cos(theta3), -np.sin(theta3)],
                                    [np.sin(theta3), np.cos(theta3)]])
        v_rotated = np.dot(rotation_matrix, velocity_vector)
        newpt = pt + v_rotated

        new_xy.append(tuple(newpt))
        new_velocity.append(tuple(v_rotated))'''

        rand_angle = random.uniform(-np.pi + 0.001, np.pi - 0.001)
        unit_vector = np.array([1,1])


        #new_theta = theta1 * rho[0] + theta2 *rho[1] + theta3*rho[2] + rand_angle * rho[3]
        if pt[0] ** 2 + pt[1] ** 2 <= radius[2]**2:
            new_velocity_vector = v_rotated_theta1 * rho[0] + v_rotated_theta2 *rho[1] + v_rotated_theta3*rho[2] + unit_vector * rho[3]
        if pt[0] ** 2 + pt[1] ** 2 > radius[2] ** 2:
            print('MEEEEEEEEE')
            new_velocity_vector = v_rotated_theta1 * -0.2 + v_rotated_theta2 *0 + v_rotated_theta3*1 + unit_vector * 0.2

            #new_velocity_vector = ab[0]*velocity_vector + ab[1]*unit_vector
        #rotation_matrix = np.array([[np.cos(new_theta), -np.sin(new_theta)],
         #                           [np.sin(new_theta), np.cos(new_theta)]])

        #v_rotated = np.dot(rotation_matrix, velocity_vector)
        newpt = pt + new_velocity_vector
        new_xy.append(tuple(newpt))
        new_velocity.append(tuple(new_velocity_vector))
        if pt[0] ** 2 + pt[1] ** 2 > radius[2]**2:
            print('\n')
            print(pt)
            print(velocity_vector)
            print(v_rotated_theta1, 'z1')
            print(v_rotated_theta2, 'z2')
            print(v_rotated_theta3, 'z3')
            print(new_velocity_vector)
            print('=================')


    xy = new_xy
    data.append(np.array(xy))
    velocity.append(new_velocity)
    #ax.scatter(centre_mass_z1[0],centre_mass_z1[1],color='orange')

#print(data)

fig, ax = plt.subplots()

ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])

print(sumz2ne)
print(sumz3ne)
print(cm)

def update(frame):
    ''' ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)'''
    ax.clear()
    '''cen = cm[frame]
    if type(cen)== np.ndarray:
        ax.scatter(cen[0], cen[1])'''

    start_points = data[frame]
    sx = start_points[:, 0]
    sy = start_points[:, 1]

    directions = velocity[frame]
    directions = np.array(directions)
    dx = directions[:, 0]
    dy = directions[:, 1]
   #print(len(start_points))
   # Update the quiver plot

    qr = ax.quiver(sx,sy, dx, dy, angles='xy', scale_units='xy', scale=1)
    #qr.set_color((random.random(), random.random(), random.random(), random.random()))

   #ax.scatter(data[frame][:, 0], data[frame][:, 1], s=15)
   # return ax

    return qr

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
'''x = [pt[0] for pt in data[0]]
y = [pt[1] for pt in data[0]]
ax.scatter(x,y,color='red')
x = [pt[0] for pt in data[1]]
y = [pt[1] for pt in data[1]]
ax.scatter(x,y,color='blue')'''


plt.show()
