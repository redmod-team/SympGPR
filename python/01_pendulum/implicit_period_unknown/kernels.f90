REAL*8 function kern_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
kern_num = exp(-0.5d0*(y_a - y_b)**2/ly**2 - 0.5d0*sin(p*(x_a - x_b))**2 &
      /lx**2)
end function
REAL*8 function dkdx_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
dkdx_num = -1.0d0*p*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin(p*(x_a &
      - x_b))**2)/(lx**2*ly**2))*sin(p*(x_a - x_b))*cos(p*(x_a - x_b))/ &
      lx**2
end function
REAL*8 function dkdy_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
dkdy_num = 1.0d0*(-y_a + y_b)*exp(0.5d0*(-lx**2*(y_a - y_b)**2 - ly**2* &
      sin(p*(x_a - x_b))**2)/(lx**2*ly**2))/ly**2
end function
REAL*8 function dkdx0_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
dkdx0_num = 1.0d0*p*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin(p*(x_a &
      - x_b))**2)/(lx**2*ly**2))*sin(p*(x_a - x_b))*cos(p*(x_a - x_b))/ &
      lx**2
end function
REAL*8 function dkdy0_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
dkdy0_num = 1.0d0*(y_a - y_b)*exp(0.5d0*(-lx**2*(y_a - y_b)**2 - ly**2* &
      sin(p*(x_a - x_b))**2)/(lx**2*ly**2))/ly**2
end function
REAL*8 function d2kdxdx0_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d2kdxdx0_num = 1.0d0*p**2*(lx**2*cos(2.0d0*p*(x_a - x_b)) - sin(p*(x_a - &
      x_b))**2*cos(p*(x_a - x_b))**2)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 &
      + ly**2*sin(p*(x_a - x_b))**2)/(lx**2*ly**2))/lx**4
end function
REAL*8 function d2kdydy0_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d2kdydy0_num = 1.0d0*(ly**2 - (y_a - y_b)**2)*exp(0.5d0*(-lx**2*(y_a - &
      y_b)**2 - ly**2*sin(p*(x_a - x_b))**2)/(lx**2*ly**2))/ly**4
end function
REAL*8 function d2kdxdy0_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d2kdxdy0_num = -1.0d0*p*(y_a - y_b)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + &
      ly**2*sin(p*(x_a - x_b))**2)/(lx**2*ly**2))*sin(p*(x_a - x_b))* &
      cos(p*(x_a - x_b))/(lx**2*ly**2)
end function
REAL*8 function d3kdxdx0dy0_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d3kdxdx0dy0_num = 1.0d0*p**2*(y_a - y_b)*(lx**2*cos(2.0d0*p*(x_a - x_b &
      )) - sin(p*(x_a - x_b))**2*cos(p*(x_a - x_b))**2)*exp(-0.5d0*(lx &
      **2*(y_a - y_b)**2 + ly**2*sin(p*(x_a - x_b))**2)/(lx**2*ly**2))/ &
      (lx**4*ly**2)
end function
REAL*8 function d3kdydy0dy0_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d3kdydy0dy0_num = (3.0d0*ly**2 - (y_a - y_b)**2)*(y_a - y_b)*exp(-0.5d0* &
      (lx**2*(y_a - y_b)**2 + ly**2*sin(p*(x_a - x_b))**2)/(lx**2*ly**2 &
      ))/ly**6
end function
REAL*8 function d3kdxdy0dy0_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d3kdxdy0dy0_num = 1.0d0*p*(1.0d0*ly**2 - (y_a - y_b)**2)*exp(-0.5d0*(lx &
      **2*(y_a - y_b)**2 + ly**2*sin(p*(x_a - x_b))**2)/(lx**2*ly**2))* &
      sin(p*(x_a - x_b))*cos(p*(x_a - x_b))/(lx**2*ly**4)
end function
REAL*8 function dkdlx_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
dkdlx_num = 1.0d0*exp(-0.5d0*(y_a - y_b)**2/ly**2 - 0.5d0*sin(p*x_a - p* &
      x_b)**2/lx**2)*sin(p*x_a - p*x_b)**2/lx**3
end function
REAL*8 function dkdly_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
dkdly_num = 1.0d0*(y_a - y_b)**2*exp(-0.5d0*(y_a - y_b)**2/ly**2 - 0.5d0 &
      *sin(p*x_a - p*x_b)**2/lx**2)/ly**3
end function
REAL*8 function d3kdxdx0dlx_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d3kdxdx0dlx_num = p**2*(-2.0d0*lx**4*cos(2.0d0*p*(x_a - x_b)) + lx**2*( &
      3.0d0*cos(2.0d0*p*(x_a - x_b)) + 2.0d0)*sin(p*(x_a - x_b))**2 - &
      1.0d0*sin(p*(x_a - x_b))**4*cos(p*(x_a - x_b))**2)*exp(-0.5d0*(lx &
      **2*(y_a - y_b)**2 + ly**2*sin(p*(x_a - x_b))**2)/(lx**2*ly**2))/ &
      lx**7
end function
REAL*8 function d3kdydy0dlx_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d3kdydy0dlx_num = 1.0d0*(ly**2 - (y_a - y_b)**2)*exp(-0.5d0*(lx**2*(y_a &
      - y_b)**2 + ly**2*sin(p*(x_a - x_b))**2)/(lx**2*ly**2))*sin(p*( &
      x_a - x_b))**2/(lx**3*ly**4)
end function
REAL*8 function d3kdxdy0dlx_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d3kdxdy0dlx_num = p*(2.0d0*lx**2 - 1.0d0*sin(p*(x_a - x_b))**2)*(y_a - &
      y_b)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin(p*(x_a - x_b)) &
      **2)/(lx**2*ly**2))*sin(p*(x_a - x_b))*cos(p*(x_a - x_b))/(lx**5* &
      ly**2)
end function
REAL*8 function d3kdxdx0dly_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d3kdxdx0dly_num = 1.0d0*p**2*(y_a - y_b)**2*(lx**2*cos(2.0d0*p*(x_a - &
      x_b)) - sin(p*(x_a - x_b))**2*cos(p*(x_a - x_b))**2)*exp(-0.5d0*( &
      lx**2*(y_a - y_b)**2 + ly**2*sin(p*(x_a - x_b))**2)/(lx**2*ly**2 &
      ))/(lx**4*ly**3)
end function
REAL*8 function d3kdydy0dly_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d3kdydy0dly_num = (-2.0d0*ly**4 + 5.0d0*ly**2*(y_a - y_b)**2 - 1.0d0*( &
      y_a - y_b)**4)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin(p*( &
      x_a - x_b))**2)/(lx**2*ly**2))/ly**7
end function
REAL*8 function d3kdxdy0dly_num(x_a, y_a, x_b, y_b, lx, ly, p)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
REAL*8, intent(in) :: p
d3kdxdy0dly_num = p*(2.0d0*ly**2 - 1.0d0*(y_a - y_b)**2)*(y_a - y_b)*exp &
      (-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin(p*(x_a - x_b))**2)/(lx &
      **2*ly**2))*sin(p*(x_a - x_b))*cos(p*(x_a - x_b))/(lx**2*ly**5)
end function
