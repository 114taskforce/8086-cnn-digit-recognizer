; =====================================================================
;  NN.ASM - 8086 handwritten digit recognizer (MASM version)
;
;  MASM port of nn_digits.asm (which was NASM syntax, "nasm -f bin").
;  This version builds with the classic MASM + LINK toolchain:
;
;     MASM NN,NN,NN;        -> NN.OBJ + NN.LST
;     LINK NN;              -> NN.EXE
;
;  Input : 28x28 binary image, 784 chars (1/#/* = ink, other = blank)
;          NN.EXE              - type 28 rows of 28 chars interactively
;          NN.EXE IMAGE.TXT    - read image from a text file
;  Net   : LeNet-style CNN, int8/int16 fixed point
;          C1  conv 6@5x5 valid  -> 24x24x6  ReLU (binary input, no mul)
;          S2  avgpool 2x2       -> 12x12x6  (sum >> 2)
;          C3  conv 16@5x5 valid -> 8x8x16   ReLU (int8*int8 imul, int32)
;          S4  avgpool 2x2       -> 4x4x16   (sum >> 2)
;          F5  dense 256->10, out = acc >> SH4, argmax (first max wins)
;
;  Prints the input as ASCII art and the recognized digit.
;  Every run is also logged to RESULTS.TXT.
;
;  NOTE: SH1/SH3/SH4 below are the real "shift constants from training"
;  (they were in nn_cfg.inc of the NASM version, provided by the user).
; =====================================================================
        .8086

; ---------------- shift constants from training (nn_cfg.inc) ---------
INCLUDE nn_cfg.inc

NC1     EQU     6                       ; C1 output channels
NC3     EQU     16                      ; C3 output channels
NOUT    EQU     10                      ; output classes
C1LEN   EQU     NC1*(2+25)              ; per channel: bias(i16)+25 taps(i8)
C3LEN   EQU     NC3*(2+150)             ; per channel: bias(i16)+150 taps(i8)
F5LEN   EQU     NOUT*(2+256)            ; per neuron:  bias(i16)+256 taps(i8)

; ---------------------------------------------------------------------
STACK_SEG SEGMENT STACK 'STACK'
        DB      4096 DUP(?)             ; 4KB stack
STACK_SEG ENDS

; ---------------------------------------------------------------------
DSEG    SEGMENT PARA PUBLIC 'DATA'

outname DB      "RESULTS.TXT",0
banner  DB      "8086 CNN DIGIT RECOGNIZER: 28x28 BINARY INPUT, LENET-STYLE INT8 NET",13,10
banner_len EQU  $-banner
instrm  DB      13,10,"TYPE 28 ROWS OF 28 CHARS:  1/#/* = INK,  ANY OTHER = BLANK",13,10
        DB      "ESC = ABORT AT ANY TIME",13,10
instrm_len EQU $-instrm
againm  DB      13,10,"ESC=QUIT, OTHER KEY=NEXT IMAGE",13,10
againm_len EQU  $-againm
opnerr  DB      "CANNOT OPEN IMAGE FILE",13,10
opnerr_len EQU  $-opnerr

; ---- weights (NN_WDAT.INC generated from net_weights.bin; the NASM
; ---- original used "incbin", which MASM does not have)
        INCLUDE NN_WDAT.INC             ; W1 / W2 / W3

; ---- variables (uninitialized) ----
h1      DB      NC1*24*24 DUP(?)        ; C1 activations (int8)
h2      DB      NC1*12*12 DUP(?)        ; S2 activations
h3      DB      NC3*8*8 DUP(?)          ; C3 activations
h4      DB      NC3*4*4 DUP(?)          ; S4 activations
curimg  DB      98 DUP(?)               ; packed current image
grid    DB      784 DUP(?)              ; input pixels 0/1
kbline  DB      28 DUP(?)               ; one row of typed chars
kbcount DW      ?                       ; chars in kbline
inname  DB      128 DUP(?)              ; image file name from command tail
filebuf DB      2048 DUP(?)
fcount  DW      ?
gcount  DW      ?
fmode   DB      ?
krow    DW      ?
fh      DW      ?
octr    DB      ?
best    DW      ?
besti   DW      ?                       ; output index (word: stores BP)
dcc     DW      ?
pk      DB      ?
kk      DW      ?
pr      DB      ?
pc      DB      ?
c1k     DB      ?
c1r     DB      ?
c1c     DB      ?
c1dr    DB      ?
c1bias  DW      ?
c1wt0   DW      ?
c1so    DW      ?
c1base  DW      ?
c3k     DB      ?
c3or    DW      ?
c3oc    DW      ?
c3bias  DW      ?
c3bhi   DW      ?
c3wt0   DW      ?
h3cur   DW      ?
linebuf DB      40 DUP(?)
artbuf  DB      848 DUP(?)
dbgbuf  DB      64 DUP(?)

DSEG    ENDS

; ---------------------------------------------------------------------
CSEG    SEGMENT PARA PUBLIC 'CODE'
        ASSUME  CS:CSEG,DS:DSEG,SS:STACK_SEG

start:
        mov     ax,DSEG
        mov     ds,ax           ; DS = data segment (ES stays = PSP)

        mov     ah,3Ch          ; create/truncate RESULTS.TXT log
        xor     cx,cx
        mov     dx,offset outname
        int     21h
        mov     fh,ax

        mov     cx,banner_len
        mov     dx,offset banner
        call    emit

        ; ---- command tail: PSP:80h = count, 81h.. = text ----
        ; (entry: ES=PSP, DS=our data segment -> read tail via ES:)
        xor     cx,cx
        mov     si,80h
        mov     cl,es:[si]
        jcxz    STKBMODE
        mov     si,81h
STSK:   mov     al,es:[si]
        cmp     al,' '
        je      STSK1
        cmp     al,9
        je      STSK1
        jmp     STFN
STSK1:  inc     si
        loop    STSK
        jmp     STKBMODE        ; tail was all blanks
STFN:   mov     di,offset inname
STFC:   mov     al,es:[si]
        cmp     al,' '
        je      STFE
        cmp     al,13
        je      STFE
        test    al,al
        je      STFE
        mov     [di],al
        inc     di
        inc     si
        loop    STFC
STFE:   mov     byte ptr [di],0
        mov     fmode,1
        jmp     STMODDONE
STKBMODE:
        mov     fmode,0
STMODDONE:

        cmp     fmode,1
        je      STFILEMODE

        ; ================= keyboard mode =================
        mov     cx,instrm_len
        mov     dx,offset instrm
        call    emit
STKBL:  call    read_kb
        jc      STEXIT          ; ESC pressed during input
        call    process
        mov     cx,againm_len
        mov     dx,offset againm
        call    emit
        mov     ah,01h          ; read key (echoed)
        int     21h
        cmp     al,27           ; ESC quits
        je      STEXIT
        jmp     STKBL

        ; ================= file mode =================
STFILEMODE:
        call    read_file
        jnc     STFOK
        mov     cx,opnerr_len
        mov     dx,offset opnerr
        call    emit
        jmp     STEXIT
STFOK:  call    process

STEXIT: mov     ah,3Eh          ; close log
        mov     bx,fh
        int     21h
        mov     ax,4C00h
        int     21h

; ---------------------------------------------------------------------
; process: show art, run CNN inference, print result
; ---------------------------------------------------------------------
process:
        call    pack_grid
        call    draw_current
        call    infer
        mov     di,offset linebuf
        mov     byte ptr [di],'P'
        mov     byte ptr [di+1],'R'
        mov     byte ptr [di+2],'E'
        mov     byte ptr [di+3],'D'
        mov     byte ptr [di+4],'I'
        mov     byte ptr [di+5],'C'
        mov     byte ptr [di+6],'T'
        mov     byte ptr [di+7],':'
        mov     byte ptr [di+8],' '
        add     al,'0'
        mov     [di+9],al
        mov     byte ptr [di+10],13
        mov     byte ptr [di+11],10
        mov     cx,12
        mov     dx,offset linebuf
        call    emit
        ret

; ---------------------------------------------------------------------
; read_kb: read 28 lines of 28 chars from keyboard into grid (0/1)
;          ESC at any time -> CF=1, caller aborts the program
; ---------------------------------------------------------------------
read_kb:
        mov     krow,0
KBROW:  mov     ax,krow
        mov     cl,28
        mul     cl              ; ax = row*28
        mov     di,ax
        add     di,offset grid
        push    di
        mov     cx,28
KBZ:    mov     byte ptr [di],0 ; default blank
        inc     di
        loop    KBZ
        pop     di
        ; manual line input (max 28 chars), ESC aborts at any time
        mov     kbcount,0
KBIN:   mov     ah,01h          ; read key with echo
        int     21h
        cmp     al,13           ; CR ends the row
        je      KBCR
        cmp     al,27           ; ESC -> quit program
        je      KBESC
        cmp     al,8            ; backspace
        je      KBBSP
        cmp     al,32           ; ignore other control chars
        jb      KBIN
        cmp     kbcount,28      ; row full: ignore extra chars
        jae     KBIN
        mov     bx,kbcount
        mov     kbline[bx],al
        inc     kbcount
        jmp     KBIN
KBBSP:  cmp     kbcount,0
        je      KBIN
        dec     kbcount
        mov     dl,8            ; erase last char on screen
        mov     ah,02h
        int     21h
        mov     dl,' '
        int     21h
        mov     dl,8
        int     21h
        jmp     KBIN
KBCR:   mov     dl,10           ; LF: next row echoes on its own line
        mov     ah,02h
        int     21h
        mov     cx,kbcount      ; apply typed chars to the grid row
        jcxz    KBDONE
        cmp     cx,28
        jbe     KBCLOK
        mov     cx,28
KBCLOK: mov     bx,offset kbline
KBCL2:  mov     al,[bx]
        inc     bx
        cmp     al,'1'
        je      KBS1
        cmp     al,'#'
        je      KBS1
        cmp     al,'*'
        je      KBS1
        jmp     KBCL3
KBS1:   mov     byte ptr [di],1
KBCL3:  inc     di
        loop    KBCL2
KBDONE: inc     krow
        cmp     krow,28
        jge     KBDONEX
        jmp     KBROW           ; jump out of short range: trampoline
KBDONEX:
        ret
KBESC:  mov     dl,13           ; CRLF so the DOS prompt starts clean
        mov     ah,02h
        int     21h
        mov     dl,10
        int     21h
        stc                     ; signal ESC to caller
        ret

; ---------------------------------------------------------------------
; read_file: read image text file named in inname into grid (0/1)
;            ignores CR/LF/space/tab, stops after 784 ink/blank chars
; ---------------------------------------------------------------------
read_file:
        mov     ah,3Dh          ; open, read-only
        xor     al,al
        mov     dx,offset inname
        int     21h
        jc      RFERR
        mov     bx,ax
        mov     ah,3Fh          ; read up to 2048 bytes
        mov     cx,2048
        mov     dx,offset filebuf
        int     21h
        mov     fcount,ax
        mov     ah,3Eh          ; close
        int     21h
        mov     si,offset filebuf
        mov     di,offset grid
        mov     cx,fcount
        mov     gcount,0
RFSC:   jcxz    RFPAD
        lodsb
        dec     cx
        cmp     al,13
        je      RFSC
        cmp     al,10
        je      RFSC
        cmp     al,9
        je      RFSC
        cmp     al,' '
        je      RFSC
        mov     byte ptr [di],0
        cmp     al,'1'
        je      RFS1
        cmp     al,'#'
        je      RFS1
        cmp     al,'*'
        je      RFS1
        jmp     RFST
RFS1:   mov     byte ptr [di],1
RFST:   inc     di
        inc     gcount
        cmp     gcount,784
        jb      RFSC
        ret                     ; got all 784 pixels
RFPAD:  cmp     di,offset grid+784
        jae     RFDONE
        mov     byte ptr [di],0
        inc     di
        jmp     RFPAD
RFDONE: ret
RFERR:  stc
        ret

; ---------------------------------------------------------------------
; pack_grid: grid 784 bytes (0/1) -> curimg 98 bytes MSB-first
;            (kept for format compatibility / debugging)
; ---------------------------------------------------------------------
pack_grid:
        mov     di,offset curimg
        mov     cx,98
        xor     al,al
PGCLR:  mov     [di],al
        inc     di
        loop    PGCLR
        mov     bx,0            ; bit index
        mov     si,offset grid
PGPK:   mov     ax,bx
        mov     cl,3
        shr     ax,cl
        mov     di,ax           ; di = byte offset
        mov     ax,bx
        and     ax,7
        mov     cl,al
        mov     dl,80h
        shr     dl,cl           ; dl = mask
        lodsb
        test    al,al
        jz      PGSK
        or      curimg[di],dl
PGSK:   inc     bx
        cmp     bx,784
        jl      PGPK
        ret

; ---------------------------------------------------------------------
; infer: CNN on grid. Returns AX = recognized digit.
; Clobbers ax,bx,cx,dx,si,di,bp.
; ---------------------------------------------------------------------
infer:
; ================= C1: conv 6@5x5 valid, binary input =================
; h1[k,r,c] = clamp((bias_k + sum w1[k,dr,dc]*grid[r+dr,c+dc]) >> SH1, 0,127)
        mov     di,offset w1
        mov     bp,offset h1
        mov     c1k,NC1
I_C1K:  mov     ax,[di]
        mov     c1bias,ax
        add     di,2
        mov     c1wt0,di
        mov     c1base,0        ; r*28+c
        mov     c1r,24
I_C1R:  mov     c1c,24
I_C1C:  mov     dx,c1bias       ; acc = bias
        mov     di,c1wt0
        mov     c1so,0          ; dr*28
        mov     c1dr,5
I_C1DR: mov     si,c1so
        add     si,c1base
        add     si,offset grid
        mov     cx,5            ; dc 0..4
I_C1DC: mov     al,[si]
        inc     si
        test    al,al
        jz      I_C1Z
        mov     al,[di]
        cbw                     ; sext tap weight
        add     dx,ax
I_C1Z:  inc     di
        dec     cx
        jnz     I_C1DC
        add     c1so,28
        dec     c1dr
        jnz     I_C1DR
        mov     cl,SH1
        sar     dx,cl
        cmp     dx,127          ; clamp to 0..127
        jle     I_C1C2
        mov     dx,127
        jmp     I_C1ST
I_C1C2: test    dx,dx
        jns     I_C1ST
        xor     dx,dx
I_C1ST: mov     [bp],dl
        inc     bp
        inc     c1base
        dec     c1c
        jnz     I_C1C
        add     c1base,4        ; next row: 28-24
        dec     c1r
        jnz     I_C1R
        ; di = bias_k+27 = bias_{k+1} already (di advances 25 taps per output)
        dec     c1k
        jz      I_C1KD
        jmp     I_C1K          ; jump out of short range: trampoline
I_C1KD:

; ================= S2: avgpool 2x2, h2 = (sum of 4) >> 2 ================
        mov     pk,NC1
        mov     kk,0
        mov     si,offset h2
I_P2K:  mov     bx,offset h1
        mov     ax,kk
        mov     cl,9
        mul     cl              ; ax = k*9
        xor     ah,ah
        mov     cl,6
        shl     ax,cl           ; ax = k*576
        add     bx,ax
        mov     pr,12
I_P2R:  mov     pc,12
I_P2C:  mov     al,[bx]
        xor     ah,ah
        mov     di,ax
        mov     al,[bx+1]
        xor     ah,ah
        add     di,ax
        add     bx,24
        mov     al,[bx]
        xor     ah,ah
        add     di,ax
        mov     al,[bx+1]
        xor     ah,ah
        add     di,ax
        sub     bx,24
        mov     dx,di
        sar     dx,1
        sar     dx,1            ; >> 2
        mov     [si],dl
        inc     si
        add     bx,2            ; next even column pair
        dec     pc
        jnz     I_P2C
        add     bx,24           ; next even row: +48, cols advanced 24
        dec     pr
        jnz     I_P2R
        inc     kk
        dec     pk
        jnz     I_P2K

; ============ C3: conv 16@5x5 valid over 6 channels, int8*int8 ==========
; h3[ko,or,oc] = clamp((sext(bias)+sum w2[ko,ci,dr,dc]*h2[ci,or+dr,oc+dc]) >> SH3, 0,127)
        mov     di,offset w2
        mov     h3cur,offset h3
        mov     c3k,NC3
I_C3K:  mov     dx,[di]         ; bias 32-bit sign extension
        xor     bx,bx
        test    dx,dx
        jns     I_C3B0
        dec     bx
I_C3B0: mov     c3bias,dx
        mov     c3bhi,bx
        add     di,2
        mov     c3wt0,di
        mov     c3or,0
I_C3OR: mov     c3oc,0
I_C3OC: mov     dx,c3bias
        mov     bx,c3bhi
        mov     bp,c3wt0
        mov     ax,c3or
        mov     cl,12
        mul     cl              ; ax = or*12
        add     ax,c3oc
        add     ax,offset h2
        mov     si,ax           ; h2 + or*12 + oc
        mov     cx,6            ; ci 0..5
I_C3CI: push    cx
        mov     cx,5            ; dr 0..4
I_C3DR: push    cx
        mov     cx,5            ; dc 0..4
I_C3DC: mov     al,[bp]         ; w2 tap
        inc     bp
        mov     ah,[si]         ; h2 value (0..127)
        inc     si
        imul    ah              ; ax = w*h
        add     dx,ax
        jnc     I_C3N1
        inc     bx
I_C3N1: test    ax,ax
        jns     I_C3N2
        dec     bx
I_C3N2: dec     cx
        jnz     I_C3DC
        add     si,7            ; next kernel row (12-5)
        pop     cx
        dec     cx
        jnz     I_C3DR
        add     si,84           ; next input channel (144-5*12)
        pop     cx
        dec     cx
        jnz     I_C3CI
        mov     cl,SH3
        jcxz    I_C3C0          ; SH=0 -> no shift
I_C3SH: sar     bx,1            ; 32-bit arithmetic shift right
        rcr     dx,1
        dec     cl
        jnz     I_C3SH
I_C3C0: cmp     dx,127          ; clamp to 0..127
        jle     I_C3C1
        mov     dx,127
        xor     bx,bx
        jmp     I_C3ST
I_C3C1: test    bx,bx
        jns     I_C3ST
        xor     dx,dx
        xor     bx,bx
I_C3ST: mov     bx,h3cur        ; store via h3 cursor (di holds w2 base)
        mov     [bx],dl
        inc     h3cur
        inc     c3oc
        cmp     c3oc,8
        jge     I_C3OCX
        jmp     I_C3OC          ; jump out of short range: trampoline
I_C3OCX:
        mov     c3oc,0
        inc     c3or
        cmp     c3or,8
        jge     I_C3ORX
        jmp     I_C3OR          ; jump out of short range: trampoline
I_C3ORX:
        add     di,150          ; di sits at bias+2; +150 -> next channel bias
        dec     c3k
        jz      I_C3KX
        jmp     I_C3K           ; jump out of short range: trampoline
I_C3KX:

; ================= S4: avgpool 2x2, h4 = (sum of 4) >> 2 ================
        mov     pk,NC3
        mov     kk,0
        mov     si,offset h4
I_P4K:  mov     bx,offset h3
        mov     ax,kk
        mov     cl,4
        mul     cl              ; ax = ko*4
        xor     ah,ah
        mov     cl,4
        shl     ax,cl           ; ax = ko*64
        add     bx,ax
        mov     pr,4
I_P4R:  mov     pc,4
I_P4C:  mov     al,[bx]
        xor     ah,ah
        mov     di,ax
        mov     al,[bx+1]
        xor     ah,ah
        add     di,ax
        add     bx,8
        mov     al,[bx]
        xor     ah,ah
        add     di,ax
        mov     al,[bx+1]
        xor     ah,ah
        add     di,ax
        sub     bx,8
        mov     dx,di
        sar     dx,1
        sar     dx,1            ; >> 2
        mov     [si],dl
        inc     si
        add     bx,2            ; next even column pair
        dec     pc
        jnz     I_P4C
        add     bx,8            ; next even row: +16, cols advanced 8
        dec     pr
        jnz     I_P4R
        inc     kk
        dec     pk
        jnz     I_P4K

; ================= F5: dense 256 -> 10, argmax first-max-wins ===========
        mov     best,8000h
        mov     di,offset w3
        mov     octr,NOUT
        xor     bp,bp           ; output index
I_F5N:  mov     dx,[di]         ; acc32 = sext(bias)
        xor     bx,bx
        test    dx,dx
        jns     I_F5B
        dec     bx
I_F5B:  add     di,2
        mov     si,offset h4
        mov     cx,256
I_F5T:  mov     al,[si]         ; h4 (0..127)
        inc     si
        mov     ah,[di]         ; w3 (int8)
        inc     di
        imul    ah              ; ax = h4*w3
        add     dx,ax
        jnc     I_F5N1
        inc     bx
I_F5N1: test    ax,ax
        jns     I_F5N2
        dec     bx
I_F5N2: dec     cx
        jnz     I_F5T
        mov     cl,SH4
        jcxz    I_F5D           ; SH=0 -> no shift
I_F5S:  sar     bx,1            ; 32-bit arithmetic shift right
        rcr     dx,1
        dec     cl
        jnz     I_F5S
I_F5D:  cmp     dx,best         ; argmax, first max wins
        jle     I_F5NB
        mov     best,dx
        mov     besti,bp
I_F5NB: inc     bp
        dec     octr
        jnz     I_F5N
        mov     ax,besti

IFDEF   DBGDUMP
        mov     si,offset w2
        call    dump16
        mov     si,offset h1+576
        call    dump16
        mov     si,offset h2
        call    dump16
        mov     si,offset h2+144
        call    dump16
        mov     si,offset h3
        call    dump16
        mov     si,offset h4
        call    dump16
ENDIF
        ret

IFDEF DBGDUMP
; ---------------------------------------------------------------------
; dump32acc: emit BX:DX acc as 8 hex digits + CRLF (debug)
; ---------------------------------------------------------------------
dump32acc:
        push    di
        push    bx
        push    dx
        mov     di,offset dbgbuf
        mov     ax,bx
        call    DA_H4
        mov     ax,dx
        call    DA_H4
        mov     byte ptr [di],13
        mov     byte ptr [di+1],10
        add     di,2
        mov     cx,10
        mov     dx,offset dbgbuf
        call    emit
        pop     dx
        pop     bx
        pop     di
        ret
DA_H4:  mov     cx,4
DA_H4L: rol     ax,1
        rol     ax,1
        rol     ax,1
        rol     ax,1
        mov     bp,ax
        and     al,15
        cmp     al,10
        jb      DA_H4D
        add     al,7
DA_H4D: add     al,'0'
        mov     [di],al
        inc     di
        mov     ax,bp
        dec     cx
        jnz     DA_H4L
        ret

; ---------------------------------------------------------------------
; dump16: emit 16 bytes at SI as hex + CRLF (debug)
; ---------------------------------------------------------------------
dump16:
        mov     di,offset dbgbuf
        mov     cx,16
D16DL:  mov     al,[si]
        inc     si
        mov     ah,al
        shr     al,1
        shr     al,1
        shr     al,1
        shr     al,1
        cmp     al,10
        jb      D16D1
        add     al,7
D16D1:  add     al,'0'
        mov     [di],al
        inc     di
        mov     al,ah
        and     al,15
        cmp     al,10
        jb      D16D2
        add     al,7
D16D2:  add     al,'0'
        mov     [di],al
        inc     di
        mov     byte ptr [di],' '
        inc     di
        loop    D16DL
        mov     byte ptr [di],13
        mov     byte ptr [di+1],10
        mov     cx,16*3+2
        mov     dx,offset dbgbuf
        call    emit
        ret
ENDIF

; ---------------------------------------------------------------------
; draw_current: emit 28x28 ASCII art of grid ('#'=ink '.'=bg)
; ---------------------------------------------------------------------
draw_current:
        mov     si,offset grid
        mov     di,offset artbuf
        mov     dcc,0
DC_D1:  mov     al,'#'
        cmp     byte ptr [si],0
        jne     DC_D2
        mov     al,'.'
DC_D2:  mov     [di],al
        inc     di
        inc     si
        inc     dcc
        cmp     dcc,28
        jl      DC_D1
        mov     dcc,0
        mov     byte ptr [di],13
        mov     byte ptr [di+1],10
        add     di,2
        cmp     si,offset grid+784
        jl      DC_D1
        mov     cx,di
        sub     cx,offset artbuf
        mov     dx,offset artbuf
        call    emit
        ret

; ---------------------------------------------------------------------
; emit: write CX bytes at DS:DX to RESULTS.TXT and to stdout
; ---------------------------------------------------------------------
emit:   push    bx
        mov     bx,fh
        mov     ah,40h
        int     21h
        mov     bx,1            ; stdout
        mov     ah,40h
        int     21h
        pop     bx
        ret

CSEG    ENDS
        END     start
