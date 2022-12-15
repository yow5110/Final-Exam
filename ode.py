def move_Euler(f, y, dt):
    ydot = f(y)
    y = y + ydot * dt
    return y

def move_RK2(f, y, dt):
    ydot = f(y)
    y_half = y + ydot * dt/2
    ydot_half= f(y_half)
    y = y + ydot_half * dt
    return y
