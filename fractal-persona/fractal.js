const FractalEngine = (() => {
  let gl, canvas;
  let progFractal, progBlur, progComp;
  let quadBuf;
  let fboF, fboB1, fboB2; // fractal FBO, bloom ping-pong
  let W, H, BW, BH;
  let startTime = performance.now();

  const state = {
    cx: 0.0, cy: 0.0,
    zoom: 1.0, rotation: 0.0,
    hue: 0.0, mode: 0.0, // 0=Newton, 1=Julia
  };

  const VERT = `
    attribute vec2 a_pos;
    void main() { gl_Position = vec4(a_pos,0.0,1.0); }
  `;

  const FRAG_FRACTAL = `
    precision highp float;
    uniform vec2  u_res;
    uniform float u_time;
    uniform vec2  u_c;
    uniform float u_zoom;
    uniform float u_rot;
    uniform float u_hue;
    uniform float u_mode;

    vec2 cmul(vec2 a,vec2 b){ return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
    vec2 cdiv(vec2 a,vec2 b){ float d=dot(b,b); return vec2(dot(a,b),a.y*b.x-a.x*b.y)/d; }

    vec3 hsv(float h,float s,float v){
      vec3 p=abs(fract(vec3(h)+vec3(1.0,2.0/3.0,1.0/3.0))*6.0-vec3(3.0));
      return v*mix(vec3(1.0),clamp(p-vec3(1.0),0.0,1.0),s);
    }

    // root colour: azul, vermelho, ciano, dourado — cycle com tempo
    vec3 rootColor(float rH){
      float h=fract(u_hue);
      if(rH<0.1)  return hsv(fract(0.62+h),1.0,1.0); // azul elétrico
      if(rH<0.3)  return hsv(fract(0.97+h),1.0,1.0); // vermelho carmesim
      if(rH<0.6)  return hsv(fract(0.50+h),1.0,1.0); // ciano
                  return hsv(fract(0.13+h),1.0,1.0); // dourado
    }

    vec3 newton(vec2 z){
      float iter=0.0;
      float trapCirc=1e9; // distância ao círculo unitário
      float trapCross=1e9; // distância aos eixos (cria padrão de cruz)
      for(int i=0;i<80;i++){
        vec2 z2=cmul(z,z);
        vec2 z3=cmul(z2,z);
        vec2 z4=cmul(z2,z2);
        vec2 num=z4-vec2(1.0,0.0)+u_c*0.35;
        vec2 den=4.0*z3;
        if(dot(den,den)<1e-12) break;
        z-=cdiv(num,den);
        iter+=1.0;
        trapCirc =min(trapCirc,  abs(length(z)-1.0));
        trapCross=min(trapCross, min(abs(z.x), abs(z.y)));
        float d0=length(z-vec2( 1.0,0.0));
        float d1=length(z-vec2( 0.0,1.0));
        float d2=length(z-vec2(-1.0,0.0));
        float d3=length(z-vec2( 0.0,-1.0));
        if(min(min(d0,d1),min(d2,d3))<0.0003) break;
      }
      float d0=length(z-vec2( 1.0,0.0));
      float d1=length(z-vec2( 0.0,1.0));
      float d2=length(z-vec2(-1.0,0.0));
      float d3=length(z-vec2( 0.0,-1.0));
      float m =min(min(d0,d1),min(d2,d3));
      float rH=(d0<=m)?0.0:(d1<=m)?0.25:(d2<=m)?0.5:0.75;

      float t    = iter/80.0;
      vec3 rCol  = rootColor(rH);
      vec3 rCol2 = rootColor(fract(rH+0.5)); // cor complementar para a cruz

      // fundo escuro com cor subtil
      vec3 col = rCol * 0.12 * (1.0-t*0.6);

      // bordo brilhante → bloom cria néon
      float edge = pow(t, 3.5) * 4.5;
      col += mix(rCol, vec3(1.0), 0.5) * edge;

      // anel do círculo unitário
      col += rCol  * exp(-trapCirc  * 20.0) * 1.1;

      // desenho de cruz — cor complementar, linhas finas
      col += rCol2 * exp(-trapCross * 30.0) * 0.7;

      return col;
    }

    vec3 julia(vec2 z){
      float iter=0.0;
      float trapPt=1e9; // point trap — cria estrelas/rosetas
      for(int i=0;i<256;i++){
        if(dot(z,z)>4.0) break;
        z=vec2(z.x*z.x-z.y*z.y+u_c.x, 2.0*z.x*z.y+u_c.y);
        iter+=1.0;
        trapPt=min(trapPt, length(z)); // distância à origem
      }
      if(iter>=256.0) return vec3(0.0);

      // smooth colouring
      float log_zn=log(dot(z,z))*0.5;
      float nu=log(log_zn/log(2.0))/log(2.0);
      float s=fract((iter+1.0-nu)*0.022);

      // paleta azul-vermelho-magenta
      float t=fract(s+u_hue);
      vec3 a=vec3(0.45,0.25,0.55);
      vec3 b=vec3(0.55,0.45,0.55);
      vec3 c=vec3(1.5,1.0,0.5);
      vec3 d=vec3(0.60,0.20,0.50);
      vec3 col=(a+b*cos(6.28318*(c*t+d)))*1.8;

      // point trap → rosetas brilhantes
      col += hsv(fract(0.62+u_hue),1.0,1.0) * exp(-trapPt*2.5) * 1.4;

      return col;
    }

    void main(){
      vec2 uv=(gl_FragCoord.xy-u_res*0.5)/min(u_res.x,u_res.y);
      uv*=2.4/u_zoom;
      float cr=cos(u_rot),sr=sin(u_rot);
      uv=vec2(uv.x*cr-uv.y*sr,uv.x*sr+uv.y*cr);
      vec3 col=(u_mode<0.5)?newton(uv):julia(uv);
      gl_FragColor=vec4(col,1.0);
    }
  `;

  // Separable Gaussian blur — direction via u_dir
  const FRAG_BLUR = `
    precision mediump float;
    uniform sampler2D u_tex;
    uniform vec2 u_res;
    uniform vec2 u_dir;
    uniform float u_bright_thresh;

    void main(){
      vec2 uv=gl_FragCoord.xy/u_res;
      vec2 off=u_dir/u_res;
      vec3 col=
        texture2D(u_tex,uv         ).rgb*0.1964+
        texture2D(u_tex,uv+off*1.5 ).rgb*0.1746+
        texture2D(u_tex,uv-off*1.5 ).rgb*0.1746+
        texture2D(u_tex,uv+off*3.5 ).rgb*0.1210+
        texture2D(u_tex,uv-off*3.5 ).rgb*0.1210+
        texture2D(u_tex,uv+off*6.0 ).rgb*0.0648+
        texture2D(u_tex,uv-off*6.0 ).rgb*0.0648+
        texture2D(u_tex,uv+off*9.0 ).rgb*0.0214+
        texture2D(u_tex,uv-off*9.0 ).rgb*0.0214;
      // bright-pass on first blur only (high threshold = only edges bloom)
      if(u_bright_thresh>0.0){
        float luma=dot(col,vec3(0.299,0.587,0.114));
        col*=smoothstep(u_bright_thresh,u_bright_thresh+0.2,luma);
      }
      gl_FragColor=vec4(col,1.0);
    }
  `;

  // Composite: fractal + bloom → screen
  const FRAG_COMP = `
    precision mediump float;
    uniform sampler2D u_fractal;
    uniform sampler2D u_bloom;
    uniform vec2 u_res;

    void main(){
      vec2 uv=gl_FragCoord.xy/u_res;
      vec3 f=texture2D(u_fractal,uv).rgb;
      vec3 b=texture2D(u_bloom,  uv).rgb;
      // additive bloom — néon intenso
      vec3 col=f + b*1.8;
      // filmic exposure (não esmaga as cores)
      col=vec3(1.0)-exp(-col*1.3);
      gl_FragColor=vec4(col,1.0);
    }
  `;

  // ---------- GL helpers ----------

  function compile(type,src){
    const s=gl.createShader(type);
    gl.shaderSource(s,src);
    gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))
      console.error('shader err:',gl.getShaderInfoLog(s));
    return s;
  }
  function link(fs){
    const p=gl.createProgram();
    gl.attachShader(p,compile(gl.VERTEX_SHADER,VERT));
    gl.attachShader(p,compile(gl.FRAGMENT_SHADER,fs));
    gl.linkProgram(p);
    return p;
  }
  function makeFBO(w,h){
    const fbo=gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
    const tex=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,tex);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    return {fbo,tex,w,h};
  }
  function drawQuad(prog,uniforms,targetFBO,vw,vh){
    gl.useProgram(prog);
    const loc=gl.getAttribLocation(prog,'a_pos');
    gl.bindBuffer(gl.ARRAY_BUFFER,quadBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
    // set uniforms
    for(const[k,v] of Object.entries(uniforms)){
      const u=gl.getUniformLocation(prog,k);
      if(!u&&u!==0) continue;
      if(typeof v==='number') gl.uniform1f(u,v);
      else if(v.length===2) gl.uniform2f(u,v[0],v[1]);
      else if(Number.isInteger(v)) gl.uniform1i(u,v);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFBO||null);
    gl.viewport(0,0,vw,vh);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  }

  function bindTex(unit,tex){
    gl.activeTexture(gl.TEXTURE0+unit);
    gl.bindTexture(gl.TEXTURE_2D,tex);
  }

  function resize(){
    W=canvas.width =window.innerWidth;
    H=canvas.height=window.innerHeight;
    BW=Math.floor(W/2);
    BH=Math.floor(H/2);
    fboF =makeFBO(W, H);
    fboB1=makeFBO(BW,BH);
    fboB2=makeFBO(BW,BH);
  }

  function init(canvasEl){
    canvas=canvasEl;
    const opts={preserveDrawingBuffer:true,antialias:false};
    gl=canvas.getContext('webgl',opts)||canvas.getContext('experimental-webgl',opts);
    progFractal=link(FRAG_FRACTAL);
    progBlur   =link(FRAG_BLUR);
    progComp   =link(FRAG_COMP);
    quadBuf=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
    resize();
    window.addEventListener('resize',resize);
  }

  function render(){
    const t=(performance.now()-startTime)/1000;

    // 1. Render fractal → fboF
    drawQuad(progFractal,{
      u_res:[W,H], u_time:t,
      u_c:[state.cx,state.cy],
      u_zoom:state.zoom, u_rot:state.rotation,
      u_hue:state.hue, u_mode:state.mode,
    }, fboF.fbo, W, H);

    // 2. Bright-pass + horizontal blur → fboB1 (half res)
    bindTex(0,fboF.tex);
    drawQuad(progBlur,{
      u_tex:0, u_res:[BW,BH],
      u_dir:[1,0], u_bright_thresh:0.55,
    }, fboB1.fbo, BW, BH);

    // 3. Vertical blur → fboB2
    bindTex(0,fboB1.tex);
    drawQuad(progBlur,{
      u_tex:0, u_res:[BW,BH],
      u_dir:[0,1], u_bright_thresh:-1.0, // skip bright-pass on second blur
    }, fboB2.fbo, BW, BH);

    // 4. Composite fractal + bloom → screen
    bindTex(0,fboF.tex);
    bindTex(1,fboB2.tex);
    const compLoc=gl.getUniformLocation(progComp,'u_fractal');
    const bloomLoc=gl.getUniformLocation(progComp,'u_bloom');
    gl.useProgram(progComp);
    gl.uniform1i(compLoc,0);
    gl.uniform1i(bloomLoc,1);
    const loc=gl.getAttribLocation(progComp,'a_pos');
    gl.bindBuffer(gl.ARRAY_BUFFER,quadBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
    gl.uniform2f(gl.getUniformLocation(progComp,'u_res'),W,H);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    gl.viewport(0,0,W,H);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  }

  function loop(){
    render();
    requestAnimationFrame(loop);
  }

  function start(canvasEl){
    init(canvasEl);
    loop();
  }

  function update(params){
    const l=(a,b,t)=>a+(b-a)*t;
    if(params.cx        !==undefined) state.cx       =l(state.cx,       params.cx,       0.14);
    if(params.cy        !==undefined) state.cy       =l(state.cy,       params.cy,       0.14);
    if(params.zoom      !==undefined) state.zoom     =l(state.zoom,     params.zoom,     0.12);
    if(params.rotation  !==undefined) state.rotation =l(state.rotation, params.rotation, 0.10);
    if(params.hue       !==undefined) state.hue      =l(state.hue,      params.hue,      0.04);
    if(params.mode      !==undefined) state.mode     =l(state.mode,     params.mode,     0.06);
  }

  function screenshot() {
    render(); // garantir frame actual
    const link = document.createElement('a');
    link.download = `fractal-persona-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  }

  return { start, update, state, screenshot };
})();
