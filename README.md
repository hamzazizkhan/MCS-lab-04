# TASK 1

In task 1 we are supposed to apply 3 effects to a boid: repulsion in zone 1, alignment in zone 2, attraction in zone 3 and create a movie of boids moving flock.
The boids movie is boidsmovie.mov (screen recording)

## content
First, functions for the three effects were created. All three functions return an angle.

z1 - repulsion effect. 

z2 - alignment

z3 - attraction

Next, data is intialised and the main loop is run where the points in each zone for each boid are obtained.

centre of mass of zone 1 and zone 3 and the velocities of neighbouring boids in zone 2 are also obtained in the main loop.

The three effects are then applied on the boid and the new velocity and new points are obtained.

## ZONE 1 (z1)
z1 takes the boids point, centre of mass in zone 1 and velocity as arguments

In z1 we first get the attraction vector which is the vector pointing to the centre of mass of particles in zone 1. 

Then we use np.arctan2() to get the angle between the boids velocity vector and the attraction vector. This gives theta1. 

In the main for loop for generating boids in each frame (main loop), the velocity vector is then rotated towards the attraction vector by getting the dot product of the rotation matrix (with angle theta1)
with the velocity vector. 

This aligns the boids velocity vector with the attraction vector.

We then negate the rotated velocity vector so that it points away from the centre of mass in zone 1 giving the final rotated velocity vector for z1 effect.

## ZONE 2 (z2)
z2 takes the boids neighbours velocities as an argument

The angle of each of the neighbourhood velocities is obtained using np.arctan2() and the average of these angles is returned as theta2.

In the main function, similar to zone 1, the velocity vector is then rotated by angle theta2 by getting the dot product of the rotation matrix (with angle theta2)
with the boids velocity vector. 

This rotates the boids velocity vector by the mean angle of its neighbouring boids giving the rotated velocity vector for z2 effect.

## ZONE 3 (z3)
The z3 function is identical to z1.

The only difference from z1 is in the main loop; the rotated velocity vector is not negated, keeping it pointed towards the centre of mass of particles in zone 3


## MAIN LOOP
Before the main loop the data frame is initiated with random data points and random normalized velocities.
The main loop runs for 100 frames. For each frame we get the velocity vector, zone 1, zone 2 and zone 3 points.

zone 1 points are obtained by : (k - d[0]) ** 2 + (v - d[1]) ** 2) <= radius[0] ** 2
where k and v are points in the data frame and d is the current boid.
zone 2 points are obtained in the same way changing the radius to zone 2 radius.
the same is done for zone 3.

The above gives us the 3 zones for the current boid.

The zone 1 points are used to calculate the centre of mass of zone 1 by averaging all the points in zone 1. The centre of mass is used in the function z1 as described in ZONE 1.
Zone 2 points are used to get the velocities of each point in zone 2 which is then used in z2 function as described under ZONE 2.
Zone 3 points are used to get the centre of mass of zone 3 similar to zone 1. Its use is described under ZONE 3.

The three effects are applied on the boid by calling z1, z2 and z3 with its respective arguments, producing a rotated velocity vector for each effect.

For each zone the case is considered that there are no points in the zone. If there are no points in the zone, the rotated vector is returned as 0, nullifying the zones effect.

After all 3 effects are applied on the boid producing 3 rotated velocity vectors, the new velocity is calculated by summing the rotated velocity vectors multiplied by its respective rho. A random rotated vector is also added to this.

The velocity is updated by taking the old velocity plus the new velocity. This is normalized by dividing the updated velocity by its magnitude.

The new point is obtained by adding the updated velocity to the point for which all the effects were applied.

If there are no points in zone 3 we take the velocity vector of the boid and add it to a random rotated vector and multpily the velocity vector and the random rotated vector by alpha and beta respectively.
The updated velocity and new point are calculated as above.

After we have obtained the new point and the new velocities for all boids, the data frame and velocity frame are updated with the new values.

To visualize the boids for each frame we use animation.FuncAnimation().


# TASK 2
agg_measure_vs_r1,r3_and_r2,r3.png = plot of aggregation vs x = r1/r3 and y = r2/r3.

rho1.png = aggregation vs rho[0]

## MEASURE OF AGGREGATION
i chose to get the euccledian distance of a boid and its neighbours in zone 3 and average the distances giving the average distance
of a boid with its neighbours. Get the average of this value for all boids to get the average distance of a boids neighbours for all boids.
high aggregation will mean that the boids are closer together and so the distance between the boid and its neighbours will be smaller.

i chose eucledian distance between boids in zone 3 so that we can see how the distance of boids within each group changes as aggregation changes. we can see this effect below:

ad = 5.212355101326541 for rho = [0.4,0.3,0.2,0.1]

ad = 5.4419098518436 for rho = [0.6,0.1,0.2,0.1]

ad = 6.159537265621044 for rho = [0.7,0.1,0.1,0.1]

ad = aggregation and rho = ρ1 +ρ2 +ρ3 +ρ4 =1.

z1 effect is increased by increasing rho[0] from 0.4 to 0.7. this provides more repulsion within groups reducing the aggregation.

as you reduce aggregation, the average distance between boids in a group increases concluding that the measure works as desired.

## 2D PLOTS
radius 2,3,8 was used for z1, z2, z3 repectively.

please refer to agg_measure_vs_r1,r3_and_r2,r3.png.

r1 = [2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

r2 = [3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]

as the radii of r1 and r2 increase, the distance between the neighouring boids also increase because the radius of repulsion increases which prevents boids from getting too close.


There are phase transitions, from 2nd to 3rd point, 4th to 5th points. one would expect aggregation measure to increase consitently.
phase transitions can be an indication of reliability of increasing r1 and r2 to decrease aggregation. up unitl a radius of 4.5 and 5.5 for r1 and r2,
the prescence of phase transitions suggests that increasing r1 and r2 is not a reliable measure to decrease aggregation. However as r1 gets suffiecently close to r3 (at r1=5.5 and r3=8)
the aggreagtion reduces consistently suggesting that increasing r1 (and r2 as r2>r1) only reliably reduces aggregation if it is increased to a point where it is sufficently close to r3.

The jumps can be explained by some boids having no points as the r1 and r2 increase. This produces a high distance value (at the value of r3 or greater)
reducing aggregation.
The dips can be explained by the fact that the radius of alignment increases, so more boids have aligned in the same general direction and so there may be less boids with no neighbours.

## 1D PLOTS
radius 2,3,8 was used for z1, z2, z3 repectively.

for rho1.png, rho1 was steadily increased while reducing rho2, 3 and 4 as shown below:

5.268873315561877  rho = [0.25,0.25,0.25,0.25]

4.862586572367124  rho = [0.3,0.2,0.25,0.25]

5.754399760593815  rho = [0.4,0.2,0.2,0.2]

4.950860126377452  rho = [0.5,0.1,0.2,0.2]

5.264569045313615  rho = [0.6,0.1,0.1,0.2]

6.159537265621044  rho = [0.7,0.1,0.1,0.1]

5.661649797697571  rho = [0.8,0.1,0.1,0]

6.584721828283921  rho = [0.9,0.1,0,0]

6.901538659033054  rho = [1,0,0,0]

the first value is the aggregation measure.

as expected, aggregation reduces as repulsion effect increases. Even though there are some jumps in the measure of aggregation, the general trend when increasing rho 1 leads to a reduction in aggregation
concluding that increasing z1 effect is a reliable way to reduce aggregation.

rho2.png shows the effect of increasing z2 effect.

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

aggregation increases as alignment effect increases. there are too many jumps in the aggregation measure concluding that increasing the z2 effect is not a reliable way
of increasing the aggregation in groups of boids. This is rational as z2 only effects alignment and not the distance between boids.

rho3.png shows the effect of increasing z3 effect.

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

There is only one jump in aggregation measure. aggregation increases as zone 3 effect increases. This is rational as z3 attracts nearby boids to the centre of mass in zone 3 resulting in boids being closer together.
increasing z3 effect would be the most reliable way to increase aggregation.




