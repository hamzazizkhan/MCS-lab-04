import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import matplotlib.patches as patches
'''
fix z2
z1 no pts
z2 no neigh
z3 no pts
'''

def z1(pt, center_of_mass, velocity):
    if center_of_mass is None:
        return 0
    repulsion_vector = -(center_of_mass - pt)
    '''ax.quiver(pt[0], pt[1], repulsion_vector[0], repulsion_vector[1], angles='xy', scale_units='xy', scale=1,
              color='violet')'''
    att_vector = (center_of_mass - pt)
    theta1 = np.arctan2(att_vector[1], att_vector[0]) - np.arctan2(velocity[1], velocity[0])
    return theta1

def z2(neighbourhood_velocities):
    if len(neighbourhood_velocities) == 0:
        return 0
    angles = [np.arctan2(vel_neigh[1], vel_neigh[0]) for vel_neigh in neighbourhood_velocities]

    theta2 = sum(angles)/len(angles)
    return theta2
#z3
def z3(pt, center_of_mass, velocity):
    if center_of_mass is None:
        return 0
    att_vector = (center_of_mass - pt)
    '''ax.quiver(pt[0], pt[1], att_vector[0], att_vector[1], angles='xy', scale_units='xy', scale=1,
              color='purple')  # velocity vector'''

    theta1 = np.arctan2(att_vector[1], att_vector[0]) - np.arctan2(velocity[1], velocity[0])
    #print(theta1 * 180/np.pi)
    return theta1
# a list of np.arrays for new points in each frame
data = []
# a list of lists for velocities in each frame
velocities = []

n = 20
radius = [10, 20, 30]

rho = [0.1,0.4,0.4,0.1]
alpha = 0.3
beta = 0.7
#rho = [1,1,1,1]
seed_value = 42
random.seed(seed_value)
x = [random.uniform(-radius[1], radius[1]) for _ in range(n)]
y = [random.uniform(-radius[1], radius[1]) for _ in range(n)]
xy_initial = np.array(list(zip(x, y)))

initial_velocitites = [1 / np.linalg.norm(pt) * pt for pt in xy_initial]
# initialise velocity frame
velocities.append(initial_velocitites)
seed_value = 59
random.seed(seed_value)
x = [random.uniform(-radius[1], radius[1]) for _ in range(n)]
y = [random.uniform(-radius[1], radius[1]) for _ in range(n)]
xy = np.array(list(zip(x, y)))
# initialise data frame
data.append(xy)



'''print(xy)
print(velocities)
print('\n')'''
frames = 3
for frame in range(frames):

    ##########################visulisation##########################
    fig, ax = plt.subplots()
    circle = patches.Circle((0, 0), radius=radius[0], edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    circle = patches.Circle((0, 0), radius=radius[1], edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    circle = patches.Circle((0, 0), radius=radius[2], edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    ##########################visulisation##########################
    data_newpts = []
    data_newvelos = []

    nc_z1 = np.array([(k, v) for k, v in data[frame] if k ** 2 + v ** 2 <= radius[0] ** 2])
    if len(nc_z1) != 0:
        centre_mass_z1 = np.array(sum(nc_z1) / len(nc_z1))
    if len(nc_z1) == 0:
        centre_mass_z1 = None

    nc_z2 = np.array([(k, v) for k, v in data[frame] if k ** 2 + v ** 2 <= radius[1] ** 2])

    nc_z3 = np.array([(k, v) for k, v in data[frame] if k ** 2 + v ** 2 <= radius[2] ** 2])
    if len(nc_z3) != 0:
        centre_mass_z3 = np.array(sum(nc_z3) / len(nc_z3))
    if len(nc_z3) == 0:
        centre_mass_z3 = None
    #print(velocities[frame])
    for i,d in enumerate(data[frame]):

        #velocity_vector = 1 / np.linalg.norm(d) * d
        velocity_vector = velocities[frame][i]
        #print(velocity_vector, frame)

        ##########################z1##########################
        theta1 = z1(d, centre_mass_z1, velocity_vector)
        rotation_matrix = np.array([[np.cos(theta1), -np.sin(theta1)],
                                            [np.sin(theta1), np.cos(theta1)]])
        v_rotated = np.dot(rotation_matrix, velocity_vector)
        v_rotated = -v_rotated
        #print(velocity_vector, v_rotated)


        ##########################z2 neigh not including the pt##########################
        neigh_rad = radius[0]

        #consider changing neighbourhood to not include its own pt if z2 gives issue
        neighbourhood = [point for point in nc_z2 if
                                    (point[0] - d[0]) ** 2 + (point[1] - d[1]) ** 2 <= neigh_rad**2]
        neighbourhood_velocities = [velocities[frame][i] for point in neighbourhood for i,pt in enumerate(data[frame])
                                    if tuple(point) == tuple(pt)]
        '''print(neighbourhood,'neigh')
        print(neighbourhood_velocities,'neigh vel')'''
        theta2 = z2(neighbourhood_velocities)
        #print(theta2,' theta 2')
        #velocity_vector2 = 1 / np.linalg.norm(xy[0]) * xy[1]
        #print(theta2 * 180/np.pi)
        rotation_matrix = np.array([[np.cos(theta2), -np.sin(theta2)],
                                            [np.sin(theta2), np.cos(theta2)]])
        v_rotated2 = np.dot(rotation_matrix, velocity_vector)
        #print(v_rotated2)

        ##########################z3##########################
        theta3 = z3(d, centre_mass_z3, velocity_vector)
        rotation_matrix = np.array([[np.cos(theta3), -np.sin(theta3)],
                                            [np.sin(theta3), np.cos(theta3)]])
        v_rotated3 = np.dot(rotation_matrix, velocity_vector)

        ##########################new_velo##########################
        rand_angle = random.uniform(-np.pi , np.pi )
        unit_vector = np.array([1, 1])
        rotation_matrix = np.array([[np.cos(rand_angle), -np.sin(rand_angle)],
                                    [np.sin(rand_angle), np.cos(rand_angle)]])
        v_rotated4 = np.dot(rotation_matrix, unit_vector)

        if len(nc_z3)==0:
            updated_velocity = alpha*velocity_vector + beta*v_rotated4
            newpt = d + updated_velocity
        else:
            new_velocity = v_rotated*rho[0] + v_rotated2*rho[1] + v_rotated3*rho[2] + v_rotated4*rho[3]
            updated_velocity = velocity_vector + new_velocity
            newpt = d + updated_velocity


        ##########################new data and velocities##########################
        data_newpts.append(newpt)
        data_newvelos.append(updated_velocity)

        ##########################visualisation##########################

        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.scatter(d[0], d[1] )
        ax.scatter(nc_z2[:,0], nc_z2[:,1], color='yellow' )
        ax.scatter(centre_mass_z1[0], centre_mass_z1[1], color='black')
        ax.scatter(centre_mass_z3[0], centre_mass_z3[1], color='black')
        ax.scatter(newpt[0], newpt[1], color='black')

        ax.quiver(d[0],d[1], velocity_vector[0], velocity_vector[1], angles='xy', scale_units='xy', scale=1, color='black') # velocity vector
        ax.quiver(d[0],d[1], v_rotated[0], v_rotated[1], angles='xy', scale_units='xy', scale=1, color='blue') # 1st effect z1
        ax.quiver(d[0],d[1], v_rotated2[0], v_rotated2[1], angles='xy', scale_units='xy', scale=1, color='orange') # z2
        ax.quiver(d[0],d[1], v_rotated3[0], v_rotated3[1], angles='xy', scale_units='xy', scale=1, color='red') # z3
        ax.quiver(d[0],d[1], new_velocity[0], new_velocity[1], angles='xy', scale_units='xy', scale=1, color='cyan') # new velo
        ax.quiver(d[0],d[1], updated_velocity[0], updated_velocity[1], angles='xy', scale_units='xy', scale=1, color='grey')  # velo_new_velocity

        #plt.savefig(f'{frame+1}_iteration.png')
        ##########################visulisation##########################

        '''print(centre_mass_z1, ' cm z1')
        print(centre_mass_z3, ' cm z2')'''
    ##########################new data and velocities to main data frames##########################
    data.append(np.array(data_newpts))
    velocities.append(data_newvelos)




##########################visulisation##########################
'''fig, ax = plt.subplots()

ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])

def update(frame):
    
    ax.clear()
    circle = patches.Circle((0, 0), radius=radius[0], edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    circle = patches.Circle((0, 0), radius=radius[1], edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    circle = patches.Circle((0, 0), radius=radius[2], edgecolor='black', facecolor='none')
    ax.add_patch(circle)

    start_points = data[frame]
    sx = start_points[:, 0]
    sy = start_points[:, 1]

    directions = velocities[frame]
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

ani = animation.FuncAnimation(fig, update, frames=frames, interval=500)'''



'''x = [pt[0] for pt in data[0]]
y = [pt[1] for pt in data[0]]
ax.scatter(x,y,color='red')
x = [pt[0] for pt in data[1]]
y = [pt[1] for pt in data[1]]
ax.scatter(x,y,color='blue')'''


plt.show()







'''x = [pt[0] for pt in data[0]]
y = [pt[1] for pt in data[0]]
ax.scatter(x,y,color='red')
x = [pt[0] for pt in data[1]]
y = [pt[1] for pt in data[1]]
ax.scatter(x,y,color='blue')
'''

'''def z2(neighbourhood):
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



n = 120
seed_value = 42
random.seed(seed_value)
x = [random.uniform(-9, 9) for _ in range(n)]
y = [random.uniform(-9, 9) for _ in range(n)]
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

frames = 15
radius = [3, 6, 9]
rho = [0.1,0.4,0.4,0.1]

cm = []

for frame in range(frames):
    nc_z1 = np.array([(k, v) for k, v in xy if k ** 2 + v ** 2 <= radius[0]])
    if len(nc_z1) != 0:
        centre_mass_z1 = np.array(sum(nc_z1) / len(nc_z1))
    else:
        centre_mass_z1 = None


    nc_z2 = np.array([(k, v) for k, v in xy if k ** 2 + v ** 2 <= radius[1]])

    nc_z3 = np.array([(k, v) for k, v in xy if k ** 2 + v ** 2 <= radius[2]])
    if len(nc_z3) != 0:
        centre_mass_z3 = np.array(sum(nc_z3) / len(nc_z3))
    else:
        centre_mass_z3 = None
    cm.append((centre_mass_z3))
    new_xy = []
    new_velocity = []
    for pt in xy:
        # zone1
        pt = np.array(pt)
        velocity_vector = 1 / np.linalg.norm(pt) * pt
        theta1 = z1(pt, centre_mass_z1)

#
        new_theta = theta1*rho[0]
        rotation_matrix = np.array([[np.cos(new_theta), -np.sin(new_theta)],
                                    [np.sin(new_theta), np.cos(new_theta)]])
        v_rotated = np.dot(rotation_matrix, velocity_vector)
        newpt = pt + v_rotated
        #print(v_rotated, velocity_vector, 'here')
        new_xy.append(tuple(newpt))
        new_velocity.append(tuple(v_rotated))
#
        # zone 2
        neigh_rad = 3
        neighbourhood = [(point) for point in nc_z2 if tuple(point) != tuple(pt) if
                         (point[0] - pt[0]) ** 2 + (point[1] - pt[1]) ** 2 <= neigh_rad]
        theta2 = z2(neighbourhood)
#
        rotation_matrix = np.array([[np.cos(theta2), -np.sin(theta2)],
                                    [np.sin(theta2), np.cos(theta2)]])
        v_rotated = np.dot(rotation_matrix, velocity_vector)
        newpt = pt + v_rotated

        new_xy.append(tuple(newpt))
        new_velocity.append(tuple(v_rotated))
 #
        # zone 3
        theta3 = z3(pt, centre_mass_z3)
#
        rotation_matrix = np.array([[np.cos(theta3), -np.sin(theta3)],
                                    [np.sin(theta3), np.cos(theta3)]])
        v_rotated = np.dot(rotation_matrix, velocity_vector)
        newpt = pt + v_rotated

        new_xy.append(tuple(newpt))
        new_velocity.append(tuple(v_rotated))
#

        rand_angle = random.uniform(-np.pi + 0.001, np.pi - 0.001) * rho[3]
        new_theta = theta1 * rho[0] + theta2 *rho[1] + theta3*rho[2] + rand_angle

        rotation_matrix = np.array([[np.cos(new_theta), -np.sin(new_theta)],
                                    [np.sin(new_theta), np.cos(new_theta)]])
        v_rotated = np.dot(rotation_matrix, velocity_vector)
        newpt = pt + v_rotated
        new_xy.append(tuple(newpt))
        new_velocity.append(tuple(v_rotated))

    xy = new_xy
    data.append(np.array(xy))
    velocity.append(new_velocity)
    #ax.scatter(centre_mass_z1[0],centre_mass_z1[1],color='orange')

#print(data)

fig, ax = plt.subplots()
#quiver = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1)
ax.set_xlim([-12, 12])
ax.set_ylim([-12, 12])

print(cm)
def update(frame):
    
    ax.clear()

    start_points = data[frame]
    sx = start_points[:, 0]
    sy = start_points[:, 1]

    directions = velocity[frame]
    directions = np.array(directions)
    dx = directions[:, 0]
    dy = directions[:, 1]
   #print(len(start_points))
   # Update the quiver plot
    ax.quiver(sx,sy, dx, dy, angles='xy', scale_units='xy', scale=1)

   #ax.scatter(data[frame][:, 0], data[frame][:, 1], s=15)
   # return ax

    return ax

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)


plt.show()
'''