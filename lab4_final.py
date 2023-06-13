import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from PIL import Image
import glob

import matplotlib.patches as patches
'''
normalize velocities?
whats happening when we add velocty effects to each other?

8 params - radius, rho, alpha, beta
'''

def z1(pt, center_of_mass, velocity):
    att_vector = (center_of_mass - pt)
    theta1 = np.arctan2(att_vector[1], att_vector[0]) - np.arctan2(velocity[1], velocity[0])
    return theta1

def z2(neighbourhood_velocities):

    angles = [np.arctan2(vel_neigh[1], vel_neigh[0]) for vel_neigh in neighbourhood_velocities]

    theta2 = sum(angles)/len(angles)
    return theta2
#z3
def z3(pt, center_of_mass, velocity):

    att_vector = (center_of_mass - pt)

    theta1 = np.arctan2(att_vector[1], att_vector[0]) - np.arctan2(velocity[1], velocity[0])
    return theta1


# a list of np.arrays for new points in each frame
data = []
# a list of lists for velocities in each frame
velocities = []

# r1 [2, 3, 4, 5, 6, 7]
# r2 [3, 4, 5, 6, 7]
n = 50
radius = [2, 3, 8]

# rho used for boids movie
rho = [0.35,0.25,0.3,0.1]
alpha = 0.3
beta = 0.7
#rho = [1,1,1,1]
seed_value = 42
random.seed(seed_value)
lim = 20
x = [random.uniform(-lim, lim) for _ in range(n)]
y = [random.uniform(-lim, lim) for _ in range(n)]
xy_initial = np.array(list(zip(x, y)))

initial_velocitites = [1 / np.linalg.norm(pt) * pt for pt in xy_initial]
normalized_velocities = [vel / np.linalg.norm(vel) for vel in initial_velocitites]

# initialise velocity frame
velocities.append(normalized_velocities)

seed_value = 59
random.seed(seed_value)
x = [random.uniform(-lim, lim) for _ in range(n)]
y = [random.uniform(-lim, lim) for _ in range(n)]
xy = np.array(list(zip(x, y)))
# initialise data frame
data.append(xy)

frames = 100

for frame in range(frames):

    ##########################visulisation##########################
    #fig, ax = plt.subplots()
    '''
    circle = patches.Circle((0, 0), radius=radius[0], edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    circle = patches.Circle((0, 0), radius=radius[1], edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    circle = patches.Circle((0, 0), radius=radius[2], edgecolor='black', facecolor='none')
    ax.add_patch(circle)'''
    ##########################visulisation##########################
    data_newpts = []
    data_newvelos = []

    #print(velocities[frame])
    for i,d in enumerate(data[frame]):

        velocity_vector = velocities[frame][i]

        ##########################z1##########################
        nc_z1 = np.array([(k, v) for k, v in data[frame] if tuple(d) != tuple([k,v]) if ((k - d[0]) ** 2 + (v - d[1]) ** 2) <= radius[0] ** 2])
        #print(d,nc_z1)
        #print('\n')
        if len(nc_z1) != 0:
            centre_mass_z1 = np.array(sum(nc_z1) / len(nc_z1))
            theta1 = z1(d, centre_mass_z1, velocity_vector)
            rotation_matrix = np.array([[np.cos(theta1), -np.sin(theta1)],
                                        [np.sin(theta1), np.cos(theta1)]])
            v_rotated = np.dot(rotation_matrix, velocity_vector)
            v_rotated = -v_rotated
        if len(nc_z1) == 0:
            #print(True, 'z1')
            v_rotated = 0


        #print(velocity_vector, v_rotated)


        ##########################z2 neigh not including the pt##########################
        nc_z2 = np.array([(k, v) for k, v in data[frame] if tuple([k,v]) != tuple(d) if((k-d[0]) ** 2 + (v-d[1]) ** 2) <= radius[1] ** 2])

        if len(nc_z2)!= 0:
            neighbourhood_velocities = [velocities[frame][b] for point in nc_z2 for b,pt in enumerate(data[frame])
                                     if tuple(point) == tuple(pt)]

            theta2 = z2(neighbourhood_velocities)

            rotation_matrix = np.array([[np.cos(theta2), -np.sin(theta2)],
                                                [np.sin(theta2), np.cos(theta2)]])
            v_rotated2 = np.dot(rotation_matrix, velocity_vector)
        if len(nc_z2)==0:
            #print(True, 'z2')
            v_rotated2=0

        ##########################z3##########################
        nc_z3 = np.array([(k, v) for k, v in data[frame] if tuple([k,v]) != tuple(d) if ((k - d[0]) ** 2 + (v - d[1]) ** 2) <= radius[2] ** 2])
        #print(d,nc_z3)
        if len(nc_z3) != 0:
            centre_mass_z3 = np.array(sum(nc_z3) / len(nc_z3))
            theta3 = z3(d, centre_mass_z3, velocity_vector)
            rotation_matrix = np.array([[np.cos(theta3), -np.sin(theta3)],
                                        [np.sin(theta3), np.cos(theta3)]])
            v_rotated3 = np.dot(rotation_matrix, velocity_vector)
        if len(nc_z3) == 0:
            #print(True)
            v_rotated3=0



        ##########################new_velo##########################
        rand_angle = random.uniform(-np.pi , np.pi )

        unit_vector = np.array([1, 1])
        rotation_matrix = np.array([[np.cos(rand_angle), -np.sin(rand_angle)],
                                    [np.sin(rand_angle), np.cos(rand_angle)]])
        v_rotated4 = np.dot(rotation_matrix, unit_vector)

        if len(nc_z3)==0:
            updated_velocity = alpha*velocity_vector + beta*v_rotated4
            updated_velocity = updated_velocity/np.linalg.norm(updated_velocity)
            newpt = d + updated_velocity
        else:
            new_velocity = v_rotated*rho[0] + v_rotated2*rho[1] + v_rotated3*rho[2] + v_rotated4*rho[3]
            updated_velocity = velocity_vector + new_velocity
            updated_velocity = updated_velocity / np.linalg.norm(updated_velocity)
            newpt = d + updated_velocity


        ##########################new data and velocities##########################
        data_newpts.append(newpt)
        data_newvelos.append(updated_velocity)

    ##########################new data and velocities to main data frames##########################
    data.append(np.array(data_newpts))
    velocities.append(data_newvelos)




##########################visulisation##########################
fig, ax = plt.subplots()

ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])

def update(frame):
    
    ax.clear()

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

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)

plt.show()


########################## task 2 ##########################

'''
for each boid, get neighbouring boids within zone 3.
calculate the distance between the boid and its neighbours.
get average distance for that boid.
'''
# frames is 100

agg_data = data[50]
# average_distances gives the average distance of a boid to its neighbours in zone 3
average_dsitances = []
for boid in agg_data:

    neighboids = np.array([(k, v,  np.sqrt((k - boid[0]) ** 2 + (v - boid[1]) ** 2)  )
                           for k, v in agg_data if tuple(boid) != tuple([k, v]) if
                      ((k - boid[0]) ** 2 + (v - boid[1]) ** 2) <= radius[2] ** 2])
    if len(neighboids)!=0:
        avg_dist = sum([pt[2] for pt in neighboids])/len(neighboids)
        average_dsitances.append(avg_dist)
    else:
        # if there are no neighs means low aggregation for that boid so avg dist is greater than radius[2]
        avg_dist = radius[2]
        average_dsitances.append(avg_dist)

ad = sum(average_dsitances)/len(average_dsitances)
print(ad)
'''
get the euccledian distance of a boid and its neighbours in zone 3 and average the distances giving the average distance
of a boids neighbours. 
high aggregation will mean that the boids are closer
together and so the distance between the boid and its neighbours will be smaller.
average this value for all boids to get the average distance of a boids neighbours for all boids.
i chose eucledian distance between boids in zone 3 so that we can see how the distance of boids 
within each group changes as aggregation changes.

ad = 5.212355101326541 for rho = [0.4,0.3,0.2,0.1]
ad = 5.4419098518436 for rho = [0.6,0.1,0.2,0.1]
ad = 6.159537265621044 for rho = [0.7,0.1,0.1,0.1]
as you reduce aggregation, the average distance between boids in a group
increases concluding that the measure works as desired
'''
x = []
rad0 = radius[0]
rad1 = radius[1]
y = []
while True:
    if rad0/radius[2] < 1:
        x.append(rad0)
        rad0 +=0.5
    if rad1/radius[2] < 1:
        y.append(rad1)
        rad1 +=0.5
    else:
        break



'''
r1 [2, 3, 4, 5, 6, 7]
r2 [3, 4, 5, 6, 7]
aggmeasure_rad0 = [5.212355101326541, 5.719125750994022, 6.221146595381217, 5.581050051101222, 5.763713082219339]

try with:
[2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
[3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
'''
aggmeasure_rad0 = [5.212355101326541, 4.904147522255776 ,5.719125750994022, 5.314898466739733 , 6.221146595381217, 5.576533041501332,  5.581050051101222, 5.574492648849, 5.763713082219339, 5.942054680999035]

x_vals = np.arange(0,0.5*len(aggmeasure_rad0), 0.5)
y_vals = aggmeasure_rad0

fig, ax = plt.subplots()
ax.set_xlim([x_vals[0], 5])
ax.set_ylim([4.5, 6.5])

ax.scatter(x_vals, y_vals, color='red')
ax.plot(x_vals, y_vals, color='green',linestyle = 'dashed')
plt.xlabel('r1/r3 and r2/r3')
plt.ylabel('agg measure')
'''plt.savefig('agg_measure_vs_r1,r3_and_r2,r3')
plt.show()'''

'''
as rho(i) increases what happens to the measure
rho(1)
5.268873315561877  rho = [0.25,0.25,0.25,0.25]
4.862586572367124  rho = [0.3,0.2,0.25,0.25]
5.754399760593815  rho = [0.4,0.2,0.2,0.2]
4.950860126377452  rho = [0.5,0.1,0.2,0.2]
5.264569045313615  rho = [0.6,0.1,0.1,0.2]
6.159537265621044  rho = [0.7,0.1,0.1,0.1]
5.661649797697571  rho = [0.8,0.1,0.1,0]
6.584721828283921  rho = [0.9,0.1,0,0]
6.901538659033054  rho = [1,0,0,0]

rho(2)
5.268873315561877  rho = [0.25,0.25,0.25,0.25]
5.573759676115517  rho = [0.2,0.3,0.25,0.25]
5.003608184583171  rho = [0.2,0.4,0.2,0.2]
5.4229350013581294 rho = [0.1,0.5,0.2,0.2]
4.6342646180845115 rho = [0.1,0.6,0.1,0.2]
5.518406906479246  rho = [0.1,0.7,0.1,0.1]
5.198686809660206  rho = [0.1,0.8,0.1,0]
5.1074011309935115 rho = [0.1,0.9,0,0]
4.254195454391614  rho = [0,1,0,0]

rho(3)
5.268873315561877  rho = [0.25,0.25,0.25,0.25]
5.253696487300989  rho = [0.25,0.2,0.3,0.25]
4.826155756953032  rho = [0.2,0.2,0.4,0.2]
4.21633775744711   rho = [0.2,0.1,0.5,0.2]
4.494264154910868  rho = [0.1,0.1,0.6,0.2]
2.5259457856213836 rho = [0.1,0.1,0.7,0.1]
1.9548732210024538 rho = [0.1,0.1,0.8,0]
1.64786411858181   rho = [0,0.1,.9,0]
1.3804517032445867 rho = [0,0,1,0]
'''
x_vals = np.arange(0.3,1.1, 0.1)
print(x_vals)
x_vals_new = []
for i in range(len(x_vals)+1):
    if i==0:
        x_vals_new.append(0.25)
    else:
        x_vals_new.append(np.round(x_vals[i-1], 2))


y_vals = [5.268873315561877,  4.862586572367124,  5.754399760593815,  4.950860126377452,  5.264569045313615,
6.159537265621044,  5.661649797697571,  6.584721828283921,  6.901538659033054]


fig, ax = plt.subplots()
ax.set_xlim([min(x_vals_new), max(x_vals_new)])
ax.set_ylim([min(y_vals), max(y_vals)])

ax.scatter(x_vals_new, y_vals, color='red')
ax.plot(x_vals_new, y_vals, color='green',linestyle = 'dashed')
plt.xlabel('rho 1')
plt.ylabel('agg measure')
'''plt.savefig('rho 1')

plt.show()'''

y_vals = [5.268873315561877,  5.573759676115517,  5.003608184583171,  5.4229350013581294,
4.6342646180845115, 5.518406906479246,  5.198686809660206 , 5.1074011309935115 ,4.254195454391614]

fig, ax = plt.subplots()
ax.set_xlim([min(x_vals_new), max(x_vals_new)])
ax.set_ylim([min(y_vals), max(y_vals)])

ax.scatter(x_vals_new, y_vals, color='red')
ax.plot(x_vals_new, y_vals, color='green',linestyle = 'dashed')
plt.xlabel('rho 2')
plt.ylabel('agg measure')
'''plt.savefig('rho 2')
plt.show()'''

y_vals = [5.268873315561877 , 5.253696487300989 , 4.826155756953032 , 4.21633775744711,
4.494264154910868 , 2.5259457856213836, 1.9548732210024538, 1.64786411858181,   1.3804517032445867]



fig, ax = plt.subplots()
ax.set_xlim([min(x_vals_new), max(x_vals_new)])
ax.set_ylim([min(y_vals), max(y_vals)])

ax.scatter(x_vals_new, y_vals, color='red')
ax.plot(x_vals_new, y_vals, color='green',linestyle = 'dashed')
plt.xlabel('rho 3')
plt.ylabel('agg measure')
'''plt.savefig('rho 3')
plt.show()'''



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