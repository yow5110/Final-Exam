
def move_RK2(diffeq, y, dt):
    ydot = diffeq( y )
    y_half = y + 1/2 * ydot * dt
    
    ydot_half  =  diffeq( y_half)
    y = y + ydot_half * dt
    
    return y
    

def move_Euler(diffeq, y, dt):
    ydot = diffeq( y )
    y = y+ ydot * dt
    return y