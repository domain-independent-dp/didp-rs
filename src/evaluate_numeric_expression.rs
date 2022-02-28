#[derive(Clone)]
enum RispExp {
    Symbol(String),
    Integer(i32),
    Continuous(f64),
    List(Vec<RispExp>),
    Func(fn(&[RispExp]) -> Result<RispExp, RispErr>),
}

#[derive(Debug)]
enum RispErr {
    Reason(String),
}

#[derive(Clone)]
struct RispEnv {
    data: HashMap<String, RispExp>,
}

fn tokenize(expr: String) -> Vec<String> {
    expr.replace("(", " ( ")
        .replace(")", " ) ")
        .split_whitspace()
        .map(|x| x.to_string())
        .collect()
}

fn parse<'a>(tokens: &'a [String]) -> Result<(RispExp, &'a [String]), RispErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or(RispErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "(" => read_seq(rest),
        ")" => Err(RispErr::Reason("unexpected `)`".to_string())),
        _ => Ok((parse_atom(token), rest)),
    }
}

fn read_seq<'a>(tokens: &'a [String]) -> Result<(RispExp, &'a [String]), RispErr> {
    let mut res: Vec<RispExp> = vec![];
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or(RispErr::Reason("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((RispExp::List(res), rest));
        }
        let (exp, new_xs) = parse(&xs)?;
        res.push(exp);
        xs = new_xs;
    }
}

fn parse_atom(token: &str) -> RispExp {
    let potentioal_integer: Result<i32, ParseIntError> = token.parse();
    match potentioal_integer {
        Ok(v) => RispExp::Integer(v),
        Err(_) => {
            let potentioal_continuous: Result<f64, ParseFloatError> = token.parse();
            match potentioal_continuous {
                Ok(v) => RispExp::Continuous(v),
                Err(_) => RispExp::Symbol(token.to_string().clone()),
            }
        }
    }
}

fn default_env() -> RispEnv {
    let mut data: HashMap<String, RispExp> = HashMap::new();
    data.insert(
        "+".to_string(),
        RispExp::Func(
            |args: &[RispExp]| -> Result<RispExp, RispErr> {
                let sum = parse_list_of_floats(args)?.iter().fold(0.0, |sum, a| sum + a);
                Ok(RispExp::Continuous(sum)))
            }
        )
    );
    data.insert(
        "-".to_string(),
        RispExp::Func(
            |args: &[RispExp]| -> Result<RispExp, RispErr> {
                let floats = parse_list_of_floats(args)?;
                let first = &floats.first().ok_or(RispErr::Reason("expected at least one number".to_string()))?;
                let sum_of_rest = floats[1..].iter().fold(0.0, |sum, a| sum + a);
                Ok(RispExp::Continuous(first - sum_of_rest)
            }
        )
    );

    RispEnv { data }
}

fn parse_list_of_floats(args: &[RispExp]) -> Result<Vec<f64>, RispErr> {
    args.iter().map(|x| parse_single_float(x)).collect()
}

fn parse_single_float(exp: &RispExp) -> Result<f64, RispErr> {
    match exp {
        RispExp::Continuous(num) => Ok(*num),
        _ => Err(RispErr::Reason("expected a continuous value".to_string()))
    }
}

fn eval(exp: &RispExp, env: &mut RispEnv) -> Result<RispExp, RispErr> {
    match exp {
        RispExp::Symbol(k) =>
            env.data.get(k).ok_or(RipsErr::Reason(format!("unexpected symbol k='{}'", k))).map(|x| x.clone()),
        RispExp::Integer(_) | RisExp::Continuous(_) => Ok(exp.clone()),
        RispExp::List(list) => {
            let first_form = list.first().ok_or(RispErr::Reason("expected a non-empty list").to_string())?;
            let arg_forms = &list[1..];
            let first_eval = eval(first_form, env);
            match first_eval {
                RispExp::Func(f) => {
                    let args_eval = arg_forms.iter().map(|x| eval(x, env)).collect::<Result<Vec<RispExp>, RispErr>>();
                    f(&args_eval)?
                },
                _ => Err(RispErr::Reason("first form must be a function".to_string())),
            }
        },
        RispExp::Func(_) => Err(RispErr::Reason("unexpected form".to_string())),
    }
}
