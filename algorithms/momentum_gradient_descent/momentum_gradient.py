def momentum_gradient_descent():
    w,b,eta=init_w,init_b,1.0
    prev_dw,prev_db,gamma=0,0,0.9
    for i in range(max_epochs):
        dw,db=0,0
        for x,y in zip(X_train,y_train):
            dw+=grad_w(w,b,x,y)
            db+=grad_b(w,b,x,y)
        v_w=gamma*prev_dw+eta*dw
        v_b=gamma*prev_db+eta*db
        w-=v_w
        b-=v_b
        prev_dw,prev_db=v_w,v_b
