import pybullet as p

# Bullet world setup placeholder for ground contact and gear points.
def create_world(time_step: float = 1.0 / 240.0):
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setTimeStep(time_step, physicsClientId=physics_client)
    plane_id = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_id)
    return physics_client
