!ADDITIONAL LINES ADDED BY YING SHI AFTER LINE 428 of src/94_scfcv/m_outscfcv.f90
!Read density file
open(99, file='/home/yteh/abinit-8.10.2/density_file_added/density_file.dat')
write(std_out,'(a)')'Reading density file!!!'
write(ab_out,'(a)')'Reading density file!!!'
do ifft=1,nfft
    read(99,*)rhor(ifft,1)
end do

!END OF ADDITIONAL LINES ADDED BY YING SHI