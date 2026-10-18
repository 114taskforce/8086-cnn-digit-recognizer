; =====================================================================
;  nn_digits.asm - 8086 handwritten digit recognizer (strict 8086 ISA)
;
;  Input : 28x28 binary image, 784 bits packed row-major MSB-first
;          into 98 bytes; manually entered or read from a text file
;          MANUAL INPUT, two ways:
;            nn.com             - type 28 rows of 28 chars interactively
;            nn.com IMAGE.TXT   - read image from a text file
;          Ink chars: 1 # *     Blank chars: 0 . space ...
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
;  Build : nasm -f bin nn_digits.asm -o nn.exe
;
;  EXE format: 32-byte MZ header built by hand (no linker needed). The
;  header uses hdrpara=0, so DOS loads the whole file (header included)
;  as one flat image: label addresses (= file offsets) are valid as-is
;  in memory. CS=DS=SS share the load segment; ES points at the PSP on
;  entry, so the command tail is read with ES: overrides and DS is set
;  from CS at startup. A 4KB stack is reserved past .bss and pre-set
;  via SS:SP.
; =====================================================================
cpu 8086
        org     0               ; flat image: file offsets = memory offsets

%include "nn_cfg.inc"           ; SH1 / SH3 / SH4 shift constants from training

NC1     equ 6                   ; C1 output channels
NC3     equ 16                  ; C3 output channels
NOUT    equ 10                  ; output classes
C1LEN   equ NC1*(2+25)          ; per channel: bias(i16)+25 taps(i8)
C3LEN   equ NC3*(2+150)         ; per channel: bias(i16)+150 taps(i8)
F5LEN   equ NOUT*(2+256)        ; per neuron:  bias(i16)+256 taps(i8)

; ---------------------------------------------------------------------
section .text
; Hand-built 32-byte MZ header (keeps the "nasm -f bin" workflow, no
; linker required). No relocations. hdrpara=0 tells DOS the load image
; starts at file offset 0, so the whole file (header included) is
; loaded as one flat image and label addresses (= file offsets) are
; valid as-is in memory. The header just rides along as data.
        db      'MZ'
        dw      ((FILEIMG-1) % 512)+1           ; bytes in last page
        dw      (FILEIMG+511)/512               ; pages
        dw      0                               ; relocation count
        dw      0                               ; header paragraphs: 0 -> don't skip header
        dw      MINPARA                         ; min extra paras: bss+stack
        dw      0                               ; max extra paras (= min)
        dw      0                               ; SS relative to load segment
        dw      stack_top                       ; initial SP
        dw      0                               ; checksum (ignored)
        dw      start                           ; initial IP (= file offset 32)
        dw      0                               ; initial CS relative
        dw      0x40                            ; reloc table offset (unused)
        dw      0                               ; overlay number
        times   32-($-$$) db 0                  ; pad header to 32 bytes
start:
        push    cs
        pop     ds              ; make DS point at our code/data segment
        mov     ah,3Ch          ; create/truncate RESULTS.TXT log
        xor     cx,cx
        mov     dx,outname
        int     21h
        mov     [fh],ax

        mov     cx,banner_len
        mov     dx,banner
        call    emit

        ; ---- command tail: PSP:80h = count, 81h.. = text ----
        ; (entry: ES=PSP, DS=our segment -> read tail via ES:)
        xor     cx,cx
        mov     cl,[es:80h]
        jcxz    .kbmode
        mov     si,81h
.sk:    mov     al,[es:si]
        cmp     al,' '
        je      .sk1
        cmp     al,9
        je      .sk1
        jmp     .fn
.sk1:   inc     si
        loop    .sk
        jmp     .kbmode         ; tail was all blanks
.fn:    mov     di,inname
.fc:    mov     al,[es:si]
        cmp     al,' '
        je      .fe
        cmp     al,13
        je      .fe
        test    al,al
        je      .fe
        mov     [di],al
        inc     di
        inc     si
        loop    .fc
.fe:    mov     byte [di],0
        mov     byte [fmode],1
        jmp     .moddone
.kbmode:mov     byte [fmode],0
.moddone:

        cmp     byte [fmode],1
        je      .filemode

        ; ================= keyboard mode =================
        mov     cx,instr_len
        mov     dx,instr
        call    emit
.kbl:   call    read_kb
        call    process
        mov     cx,againm_len
        mov     dx,againm
        call    emit
        mov     ah,01h          ; read key (echoed)
        int     21h
        cmp     al,27           ; ESC quits
        je      .exit
        jmp     .kbl

        ; ================= file mode =================
.filemode:
        call    read_file
        jnc     .fok
        mov     cx,opnerr_len
        mov     dx,opnerr
        call    emit
        jmp     .exit
.fok:   call    process

.exit:  mov     ah,3Eh          ; close log
        mov     bx,[fh]
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
        mov     di,linebuf
        mov     byte [di],'P'
        mov     byte [di+1],'R'
        mov     byte [di+2],'E'
        mov     byte [di+3],'D'
        mov     byte [di+4],'I'
        mov     byte [di+5],'C'
        mov     byte [di+6],'T'
        mov     byte [di+7],':'
        mov     byte [di+8],' '
        add     al,'0'
        mov     [di+9],al
        mov     byte [di+10],13
        mov     byte [di+11],10
        mov     cx,12
        mov     dx,linebuf
        call    emit
        ret

; ---------------------------------------------------------------------
; read_kb: read 28 lines of 28 chars from keyboard into grid (0/1)
; ---------------------------------------------------------------------
read_kb:
        mov     byte [kbbuf],29 ; max 28 chars + CR for DOS 0Ah
        mov     word [krow],0
.row:   mov     ax,[krow]
        mov     cl,28
        mul     cl              ; ax = row*28
        mov     di,ax
        add     di,grid
        push    di
        mov     cx,28
.z:     mov     byte [di],0     ; default blank
        inc     di
        loop    .z
        pop     di
        mov     ax,0C0Ah        ; flush typeahead + buffered line input
        mov     dx,kbbuf
        int     21h
        mov     cl,[kbbuf+1]    ; chars typed
        xor     ch,ch
        jcxz    .nr
        cmp     cl,28
        jbe     .clok
        mov     cl,28
.clok:  mov     bx,kbbuf+2
.cl2:   mov     al,[bx]
        inc     bx
        cmp     al,'1'
        je      .s1
        cmp     al,'#'
        je      .s1
        cmp     al,'*'
        je      .s1
        jmp     .cl3
.s1:    mov     byte [di],1
.cl3:   inc     di
        loop    .cl2
.nr:    inc     word [krow]
        cmp     word [krow],28
        jl      .row
        ret

; ---------------------------------------------------------------------
; read_file: read image text file named in inname into grid (0/1)
;            ignores CR/LF/space/tab, stops after 784 ink/blank chars
; ---------------------------------------------------------------------
read_file:
        mov     ah,3Dh          ; open, read-only
        xor     al,al
        mov     dx,inname
        int     21h
        jc      .err
        mov     bx,ax
        mov     ah,3Fh          ; read up to 2048 bytes
        mov     cx,2048
        mov     dx,filebuf
        int     21h
        mov     [fcount],ax
        mov     ah,3Eh          ; close
        int     21h
        mov     si,filebuf
        mov     di,grid
        mov     cx,[fcount]
        mov     word [gcount],0
.sc:    jcxz    .pad
        lodsb
        dec     cx
        cmp     al,13
        je      .sc
        cmp     al,10
        je      .sc
        cmp     al,9
        je      .sc
        cmp     al,' '
        je      .sc
        mov     byte [di],0
        cmp     al,'1'
        je      .s1
        cmp     al,'#'
        je      .s1
        cmp     al,'*'
        je      .s1
        jmp     .st
.s1:    mov     byte [di],1
.st:    inc     di
        inc     word [gcount]
        cmp     word [gcount],784
        jb      .sc
        ret                     ; got all 784 pixels
.pad:   cmp     di,grid+784
        jae     .done
        mov     byte [di],0
        inc     di
        jmp     .pad
.done:  ret
.err:   stc
        ret

; ---------------------------------------------------------------------
; pack_grid: grid 784 bytes (0/1) -> curimg 98 bytes MSB-first
;            (kept for format compatibility / debugging)
; ---------------------------------------------------------------------
pack_grid:
        mov     di,curimg
        mov     cx,98
        xor     al,al
.clr:   mov     [di],al
        inc     di
        loop    .clr
        mov     bx,0            ; bit index
        mov     si,grid
.pk:    mov     ax,bx
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
        jz      .sk
        or      [curimg+di],dl
.sk:    inc     bx
        cmp     bx,784
        jl      .pk
        ret

; ---------------------------------------------------------------------
; infer: CNN on grid. Returns AX = recognized digit.
; Clobbers ax,bx,cx,dx,si,di,bp.
; ---------------------------------------------------------------------
infer:
; ================= C1: conv 6@5x5 valid, binary input =================
; h1[k,r,c] = clamp((bias_k + sum w1[k,dr,dc]*grid[r+dr,c+dc]) >> SH1, 0,127)
        mov     di,w1
        mov     bp,h1
        mov     byte [c1k],NC1
.c1k:   mov     ax,[di]
        mov     [c1bias],ax
        add     di,2
        mov     [c1wt0],di
        mov     word [c1base],0     ; r*28+c
        mov     byte [c1r],24
.c1r:   mov     byte [c1c],24
.c1c:   mov     dx,[c1bias]         ; acc = bias
        mov     di,[c1wt0]
        mov     word [c1so],0       ; dr*28
        mov     byte [c1dr],5
.c1dr:  mov     si,[c1so]
        add     si,[c1base]
        add     si,grid
        mov     cx,5                ; dc 0..4
.c1dc:  mov     al,[si]
        inc     si
        test    al,al
        jz      .c1z
        mov     al,[di]
        cbw                     ; sext tap weight
        add     dx,ax
.c1z:   inc     di
        dec     cx
        jnz     .c1dc
        add     word [c1so],28
        dec     byte [c1dr]
        jnz     .c1dr
        mov     cl,SH1
        sar     dx,cl
        cmp     dx,127          ; clamp to 0..127
        jle     .c1c2
        mov     dx,127
        jmp     .c1st
.c1c2:  test    dx,dx
        jns     .c1st
        xor     dx,dx
.c1st:  mov     [bp],dl
        inc     bp
        inc     word [c1base]
        dec     byte [c1c]
        jnz     .c1c
        add     word [c1base],4     ; next row: 28-24
        dec     byte [c1r]
        jnz     .c1r
        ; di = bias_k+27 = bias_{k+1} already (di advances 25 taps per output)
        dec     byte [c1k]
        jnz     .c1k

; ================= S2: avgpool 2x2, h2 = (sum of 4) >> 2 ================
        mov     byte [pk],NC1
        mov     word [kk],0
        mov     si,h2
.p2k:   mov     bx,h1
        mov     ax,[kk]
        mov     cl,9
        mul     cl              ; ax = k*9
        xor     ah,ah
        mov     cl,6
        shl     ax,cl           ; ax = k*576
        add     bx,ax
        mov     byte [pr],12
.p2r:   mov     byte [pc],12
.p2c:   mov     al,[bx]
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
        add     bx,2           ; next even column pair
        dec     byte [pc]
        jnz     .p2c
        add     bx,24           ; next even row: +48, cols advanced 24
        dec     byte [pr]
        jnz     .p2r
        inc     word [kk]
        dec     byte [pk]
        jnz     .p2k

; ============ C3: conv 16@5x5 valid over 6 channels, int8*int8 ==========
; h3[ko,or,oc] = clamp((sext(bias)+sum w2[ko,ci,dr,dc]*h2[ci,or+dr,oc+dc]) >> SH3, 0,127)
        mov     di,w2
        mov     word [h3cur],h3
        mov     byte [c3k],NC3
.c3k:   mov     dx,[di]             ; bias 32-bit sign extension
        xor     bx,bx
        test    dx,dx
        jns     .c3b0
        dec     bx
.c3b0:  mov     [c3bias],dx
        mov     [c3bhi],bx
        add     di,2
        mov     [c3wt0],di
        mov     word [c3or],0
.c3or:  mov     word [c3oc],0
.c3oc:  mov     dx,[c3bias]
        mov     bx,[c3bhi]
        mov     bp,[c3wt0]
        mov     ax,[c3or]
        mov     cl,12
        mul     cl              ; ax = or*12
        add     ax,[c3oc]
        add     ax,h2
        mov     si,ax               ; h2 + or*12 + oc
        mov     cx,6                ; ci 0..5
.c3ci:  push    cx
        mov     cx,5                ; dr 0..4
.c3dr:  push    cx
        mov     cx,5                ; dc 0..4
.c3dc:  mov     al,[bp]             ; w2 tap
        inc     bp
        mov     ah,[si]             ; h2 value (0..127)
        inc     si
        imul    ah                  ; ax = w*h
        add     dx,ax
        jnc     .c3n1
        inc     bx
.c3n1:  test    ax,ax
        jns     .c3n2
        dec     bx
.c3n2:  dec     cx
        jnz     .c3dc
        add     si,7                ; next kernel row (12-5)
        pop     cx
        dec     cx
        jnz     .c3dr
        add     si,84               ; next input channel (144-5*12)
        pop     cx
        dec     cx
        jnz     .c3ci
        mov     cl,SH3
        jcxz    .c3c0               ; SH=0 -> no shift
.c3sh:  sar     bx,1                ; 32-bit arithmetic shift right
        rcr     dx,1
        dec     cl
        jnz     .c3sh
.c3c0:  cmp     dx,127          ; clamp to 0..127
        jle     .c3c1
        mov     dx,127
        xor     bx,bx
        jmp     .c3st
.c3c1:  test    bx,bx
        jns     .c3st
        xor     dx,dx
        xor     bx,bx
.c3st:  mov     bx,[h3cur]          ; store via h3 cursor (di holds w2 base)
        mov     [bx],dl
        inc     word [h3cur]
        inc     word [c3oc]
        cmp     word [c3oc],8
        jl      .c3oc
        mov     word [c3oc],0
        inc     word [c3or]
        cmp     word [c3or],8
        jl      .c3or
        add     di,150              ; di sits at bias+2; +150 -> next channel bias
        dec     byte [c3k]
        jnz     .c3k

; ================= S4: avgpool 2x2, h4 = (sum of 4) >> 2 ================
        mov     byte [pk],NC3
        mov     word [kk],0
        mov     si,h4
.p4k:   mov     bx,h3
        mov     ax,[kk]
        mov     cl,4
        mul     cl              ; ax = ko*4
        xor     ah,ah
        mov     cl,4
        shl     ax,cl           ; ax = ko*64
        add     bx,ax
        mov     byte [pr],4
.p4r:   mov     byte [pc],4
.p4c:   mov     al,[bx]
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
        add     bx,2           ; next even column pair
        dec     byte [pc]
        jnz     .p4c
        add     bx,8            ; next even row: +16, cols advanced 8
        dec     byte [pr]
        jnz     .p4r
        inc     word [kk]
        dec     byte [pk]
        jnz     .p4k

; ================= F5: dense 256 -> 10, argmax first-max-wins ===========
        mov     word [best],8000h
        mov     di,w3
        mov     byte [octr],NOUT
        xor     bp,bp               ; output index
.f5n:   mov     dx,[di]             ; acc32 = sext(bias)
        xor     bx,bx
        test    dx,dx
        jns     .f5b
        dec     bx
.f5b:   add     di,2
        mov     si,h4
        mov     cx,256
.f5t:   mov     al,[si]             ; h4 (0..127)
        inc     si
        mov     ah,[di]             ; w3 (int8)
        inc     di
        imul    ah                  ; ax = h4*w3
        add     dx,ax
        jnc     .f5n1
        inc     bx
.f5n1:  test    ax,ax
        jns     .f5n2
        dec     bx
.f5n2:  dec     cx
        jnz     .f5t
        mov     cl,SH4
        jcxz    .f5d                ; SH=0 -> no shift
.f5s:   sar     bx,1                ; 32-bit arithmetic shift right
        rcr     dx,1
        dec     cl
        jnz     .f5s
.f5d:   cmp     dx,[best]           ; argmax, first max wins
        jle     .f5nb
        mov     [best],dx
        mov     [besti],bp
.f5nb:  inc     bp
        dec     byte [octr]
        jnz     .f5n
        mov     al,[besti]
        mov     ah,0
%ifdef DBGDUMP
        mov     si,w2
        call    dump16
        mov     si,h1+576
        call    dump16
        mov     si,h2
        call    dump16
        mov     si,h2+144
        call    dump16
        mov     si,h3
        call    dump16
        mov     si,h4
        call    dump16
%endif
        ret

%ifdef DBGDUMP
; ---------------------------------------------------------------------
; dump32acc: emit BX:DX acc as 8 hex digits + CRLF (debug)
; ---------------------------------------------------------------------
dump32acc:
        push    di
        push    bx
        push    dx
        mov     di,dbgbuf
        mov     ax,bx
        call    .h4
        mov     ax,dx
        call    .h4
        mov     byte [di],13
        mov     byte [di+1],10
        add     di,2
        mov     cx,10
        mov     dx,dbgbuf
        call    emit
        pop     dx
        pop     bx
        pop     di
        ret
.h4:    mov     cx,4
.h4l:   rol     ax,1
        rol     ax,1
        rol     ax,1
        rol     ax,1
        mov     bp,ax
        and     al,15
        cmp     al,10
        jb      .h4d
        add     al,7
.h4d:   add     al,'0'
        mov     [di],al
        inc     di
        mov     ax,bp
        dec     cx
        jnz     .h4l
        ret
%endif

%ifdef DBGDUMP
; ---------------------------------------------------------------------
; dump16: emit 16 bytes at SI as hex + CRLF (debug)
; ---------------------------------------------------------------------
dump16:
        mov     di,dbgbuf
        mov     cx,16
.dl:    mov     al,[si]
        inc     si
        mov     ah,al
        shr     al,1
        shr     al,1
        shr     al,1
        shr     al,1
        cmp     al,10
        jb      .d1
        add     al,7
.d1:    add     al,'0'
        mov     [di],al
        inc     di
        mov     al,ah
        and     al,15
        cmp     al,10
        jb      .d2
        add     al,7
.d2:    add     al,'0'
        mov     [di],al
        inc     di
        mov     byte [di],' '
        inc     di
        loop    .dl
        mov     byte [di],13
        mov     byte [di+1],10
        mov     cx,16*3+2
        mov     dx,dbgbuf
        call    emit
        ret
%endif

; ---------------------------------------------------------------------
; draw_current: emit 28x28 ASCII art of grid ('#'=ink '.'=bg)
; ---------------------------------------------------------------------
draw_current:
        mov     si,grid
        mov     di,artbuf
        mov     word [dcc],0
.d1:    mov     al,'#'
        cmp     byte [si],0
        jne     .d2
        mov     al,'.'
.d2:    mov     [di],al
        inc     di
        inc     si
        inc     word [dcc]
        cmp     word [dcc],28
        jl      .d1
        mov     word [dcc],0
        mov     byte [di],13
        mov     byte [di+1],10
        add     di,2
        cmp     si,grid+784
        jl      .d1
        mov     cx,di
        sub     cx,artbuf
        mov     dx,artbuf
        call    emit
        ret

; ---------------------------------------------------------------------
; emit: write CX bytes at DS:DX to RESULTS.TXT and to stdout
; ---------------------------------------------------------------------
emit:   push    bx
        mov     bx,[fh]
        mov     ah,40h
        int     21h
        mov     bx,1                    ; stdout
        mov     ah,40h
        int     21h
        pop     bx
        ret

TEXTLEN equ     $-$$            ; header+.text bytes (file size before .data)

; ---------------------------------------------------------------------
section .data align=1                   ; align=1: no pad between sections
outname:db      "RESULTS.TXT",0
banner: db      "8086 CNN DIGIT RECOGNIZER: 28x28 BINARY INPUT, LENET-STYLE INT8 NET",13,10
banner_len equ  $-banner
instr:  db      13,10,"TYPE 28 ROWS OF 28 CHARS:  1/#/* = INK,  ANY OTHER = BLANK",13,10
instr_len equ   $-instr
againm: db      13,10,"ESC=QUIT, OTHER KEY=NEXT IMAGE",13,10
againm_len equ  $-againm
opnerr: db      "CANNOT OPEN IMAGE FILE",13,10
opnerr_len equ  $-opnerr

w1:     incbin  "net_weights.bin",0,C1LEN               ; C1: [bias i16][25 w i8] x6
w2:     incbin  "net_weights.bin",C1LEN,C3LEN           ; C3: [bias i16][150 w i8] x16
w3:     incbin  "net_weights.bin",C1LEN+C3LEN,F5LEN     ; F5: [bias i16][256 w i8] x10

DATALEN equ     $-$$            ; .data bytes
FILEIMG equ     TEXTLEN+DATALEN ; total file/image size (scalar, for the header)

; ---------------------------------------------------------------------
section .bss align=1            ; nobits: lives past the image in memory
h1:     resb    NC1*24*24       ; C1 activations (int8)
h2:     resb    NC1*12*12       ; S2 activations
h3:     resb    NC3*8*8         ; C3 activations
h4:     resb    NC3*4*4         ; S4 activations
curimg: resb    98              ; packed current image
grid:   resb    784             ; input pixels 0/1
kbbuf:  resb    34              ; DOS buffered input
inname: resb    128             ; image file name from command tail
filebuf:resb    2048
fcount: resw    1
gcount: resw    1
fmode:  resb    1
krow:   resw    1
fh:     resw    1
octr:   resb    1
best:   resw    1
besti:  resb    1
dcc:    resw    1
pk:     resb    1
kk:     resw    1
pr:     resb    1
pc:     resb    1
c1k:    resb    1
c1r:    resb    1
c1c:    resb    1
c1dr:   resb    1
c1bias: resw    1
c1wt0:  resw    1
c1so:   resw    1
c1base: resw    1
c3k:    resb    1
c3or:   resw    1
c3oc:   resw    1
c3bias: resw    1
c3bhi:  resw    1
c3wt0:  resw    1
h3cur:  resw    1
linebuf:resb    40
artbuf: resb    848
dbgbuf: resb    64

stack:  resb    4096            ; private stack past .bss
stack_top:
BSSSZ   equ     $-$$            ; bss+stack bytes (beyond the image)
MINPARA equ     (BSSSZ+15)/16   ; minimum extra paragraphs for the loader
